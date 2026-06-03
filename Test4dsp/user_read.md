# C6678 端到端 IR 化路径 —— 用户使用与学习指南

> 适用对象：拿到这份 TVM 仓库后，希望快速理解"用户写一段 TVMScript →
> 跑出 c6678 板上可链接的 bare-C 源码"这一条路径长成什么样、还有哪些洞
> 没填，并能照着跑通几个测试看一眼实际产出的同事 / 学习者。
>
> 阅读顺序建议：§1 用户视角 → §2 当前能输出的 C 源码长什么样 → §3 测试
> 入口与运行命令 → §4 已落地子任务一览 → §5 还差什么 → §6 想加新算子
> 怎么开局。
>
> 路线图主文档：[`learning.md`](./learning.md)（架构 + 路线图 + 现状）；
> 流水线挂载点 / pass 选址原则在 §4.1 / §4.3 两节；本文是它的"用户向"
> 子集，把分散的入口拢成一篇。

---

## 1. 用户输入：写什么、调什么

用户唯一入口：一段标准 TVMScript（`@T.prim_func`）+ `Target("c6678")` +
`tvm.tirx.build`。**不需要**自己处理 cache_read / DMA / multicore /
core_mask —— 这些全部由 c6678 专属流水线自动接管。

最小可运行例（与 `Test4dsp/tests/test_c6678_end_to_end.py` 同源）：

```python
import tvm
from tvm import s_tir
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T

M, N, K = 128, 128, 128

@T.prim_func
def matmul_fp32(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    T.func_attr({"global_symbol": "matmul_fp32", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.sblock("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
target = tvm.target.Target("c6678")
with target:
    sch_mod = dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)
runtime_mod = tvm.tirx.build(sch_mod, target=target)
print(runtime_mod.inspect_source("c6678"))    # 一份 .c 源码字符串
```

特点：

* TVMScript **不带任何 c6678 关键字**（没有 `cache_read`、没有 `parallel`、
  没有 `dma_*`），全是 ND（自然描述）。
* Target 串只填 `"c6678"` 即可；硬件常量由
  [`tvm.tirx.c6678_config`](file:///home/tangqingyun/tvm/python/tvm/tirx/c6678_config.py)
  统一兜底（A.1）。
* 需要先经过 `dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)`
  抽特征 + 选模板 + tile + cache_read + parallel + annotate；这一段属于
  schedule 阶段，由 [`dlight.c6678`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/__init__.py)
  接管。
* `tvm.tirx.build` 内部按 `default_s_tir_pipeline` 串行跑，会自动把
  c6678 专属 pass（A.2 / A.3 / A.7 / A.8 / A.9 / A.9 follow-up）挂在
  4 个固定挂载点上。流水线全图见
  [`learning.md` §4.1.2](./learning.md#412-图-b当前管线快照2026-05-24-实测)。

---

## 2. 当前能输出什么：bare-C 源码（2345 chars）

跑完 §1 的脚本，`runtime_mod.inspect_source("c6678")` 现在固定产出
**2345 字符**的 c6678 C 源码（实测快照，由
[`test_full_source_total_size_matches_baseline`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_end_to_end.py#L181-L195)
盯死，"零差分"防回归）。其要素如下：

```
void matmul_fp32(float* A, float* B, float* C, int32_t core_mask) {
    if (GetCoreNum(core_mask) > 0 && c6678_get_core_id(core_mask) >= 0) {
        float A_global_l2[32768];                 // L2 staging：A tile + B tile 合并
        for (i_outer = c6678_get_core_id(core_mask);
             i_outer < 4;
             i_outer += GetCoreNum(core_mask)) {
            for (j_outer ...) for (k_outer ...) {
                load_row_major_tile((&(A[0])), (&(A_global_l2[0])),
                                    ..., 32, 32, 128, 4, "global");
                load_row_major_tile((&(B[0])), (&(A_global_l2[16384])),
                                    ..., 32, 32, 128, 4, "global");
                /* 内层 32×32×32 reduce-style matmul（serial 三层 for） */
            }
        }
    }
    C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask));
}
```

直接观测到的 6 个特征（每一个都对应了一个已落地 pass）：

| 现象 | 出处 pass | 对应路线图 |
|---|---|---|
| `void matmul_fp32(float*, float*, float*, int32_t core_mask)` 入口 | `C6678LowerEntry` | A.7 ✅ |
| `int32_t core_mask` 形参（未在 TVMScript 出现，pass 自动追加） | `C6678MulticoreLower` | A.8 ✅ |
| `if (GetCoreNum > 0 && c6678_get_core_id >= 0) { for ... }` 多核分块 | `C6678MulticoreLower` | A.8 ✅ |
| `C6678E_SyncN(...)` 末尾同步 | `C6678MulticoreLower` | A.8 ✅ |
| `float A_global_l2[32768]` L2 staging（A tile + B tile 合并） | `C6678AnnotateL2Alloc` + StorageRewrite | A.9 ✅ |
| `load_row_major_tile(&A[0], &A_global_l2[0], ...)` DMA 调用 | `C6678DMALower` | A.9 ✅ |

**IR 层只生成"纯算术 + extern 调用"**：源码里完全不出现
`DNUM`、`LL2_*_BASE`、`#include <78NE/initial.h>` 等 BSP 头宏 / 板上常量；
所有平台细节封在 `extern` 函数后，BSP 工程链接此 `.c` 时按
[`learning.md` §4.1.3 ABI 契约表](./learning.md#413-bsp-abi-契约已固化bsp-端必须实现)
提供以下符号即可：

```c
int32_t GetCoreNum(int32_t core_mask);
int32_t c6678_get_core_id(int32_t core_mask);
void C6678E_SyncN(int32_t core_num, int32_t core_id);
void load_row_major_tile(const void* src, void* dst,
                         int32_t row0, int32_t col0,
                         int32_t rows, int32_t cols,
                         int32_t src_ld, int32_t elem_bytes,
                         const char* src_scope);
```

---

## 3. 想看实物：测试入口与运行命令

运行任何 c6678 相关 Python 测试都必须先进 conda 环境，并把
`tvm/python` + `3rdparty/tvm-ffi/python` 都塞进 `PYTHONPATH`。命令模板：

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python <要跑的测试脚本路径>
```

| 想看什么 | 跑哪个 |
|---|---|
| **端到端串一遍**（强烈推荐第一个看） | [`Test4dsp/tests/test_c6678_end_to_end.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_end_to_end.py) —— 7 个用例覆盖 A.2 / A.3 / A.7 / A.8 / A.9 + 2345 chars 零差分快照 |
| **想看 softmax 端到端**（PR-S2 phase A） | [`Test4dsp/generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) → 落盘 [`tests/generated_c6678_softmax_via_build.c`](file:///home/tangqingyun/tvm/Test4dsp/tests/generated_c6678_softmax_via_build.c)（1925 chars） |
| **softmax 形式化回归**（7 用例） | [`Test4dsp/tests/test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py) —— 覆盖 bare-C 入口 / 多核派发+SyncN / `T_max/T_exp/T_sum` 栈数组 / 4 串行内层 / `expf(` 直降 / 1925 chars 零差分 |
| 纯 demo，能直接看到 .c 文件 | [`Test4dsp/generate_c6678_matmul_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul_via_build.py) —— 跑完会落盘 [`tests/generated_c6678_matmul_via_build.c`](file:///home/tangqingyun/tvm/Test4dsp/tests/generated_c6678_matmul_via_build.c) |
| 字符串模板基线（旧路径，留作对照） | [`Test4dsp/generate_c6678_matmul.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul.py) → [`tests/generated_c6678_matmul_baseline.c`](file:///home/tangqingyun/tvm/Test4dsp/tests/generated_c6678_matmul_baseline.c) |
| A.3 单 pass 单元测试 | [`Test4dsp/tests/test_c6678_dma_legalize.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_dma_legalize.py) —— 6 用例：合法 / 非法 scope / 超容量 / 行对齐告警 / 幂等 / 非 c6678 跳过 |
| A.5 派发器单元测试 | [`Test4dsp/tests/test_c6678_dispatcher.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_dispatcher.py) |
| A.2 storage plan attrs | [`Test4dsp/tests/test_c6678_storage_plan.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_storage_plan.py) |
| 出码字节数 baseline 防回归 | [`Test4dsp/tests/probe_c6678_build_baseline.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/probe_c6678_build_baseline.py) |

---

## 4. 已落地子任务一览（路线图 §4.2）

| 编号 | 名称 | 入口文件 | 状态 |
|---|---|---|---|
| A.1 | C6678 attrs helper | [`tirx/c6678_config.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/c6678_config.py) | ✅ |
| A.2 | Storage planning pass | [`tirx/transform/c6678_storage_plan.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_storage_plan.py) | ✅（仅写 attrs，下游未消费） |
| A.3 | DMA 合法性校验 | [`tirx/transform/c6678_dma_legalize.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_legalize.py) | ✅ 紧跟 A.9 之后 |
| A.4 | 算子特征抽取 | [`tirx/analysis/c6678_features.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/analysis/c6678_features.py) | ✅ |
| A.5 | 专家模板派发器 | [`s_tir/dlight/c6678/dispatcher.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/dispatcher.py) | ✅（L2 gate 已改为保守全量 staging 估算） |
| A.6 | dlight schedule 模板 | [`s_tir/dlight/c6678/`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678) | 🟡 Matmul + Softmax phase A + ElementGreaterEqual same-shape 已接入；broadcast/conv/reduce/LSTM pattern 仍未完成 |
| A.7 | Bare-C 入口 | [`tirx/transform/c6678_lower_entry.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_lower_entry.py) | ✅ |
| A.8 | 多核派发 + SyncN | [`tirx/transform/c6678_multicore_lower.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_multicore_lower.py) | ✅ |
| A.9 | DMA Lower + L2 重写 | [`tirx/transform/c6678_dma_lower.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py) | ✅（未成功 lower 的 DMA annotation 会直接报错） |
| A.10 step1 | dlight 端 cache_read + annotate | [`s_tir/dlight/c6678/matmul.py::MatmulGemmTemplate.apply`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/dispatcher.py#L106-L160) | ✅ |
| **PR-S2 phase A** | dlight `Softmax` rule + `C6678AnnotateGlobalAlloc` pass | [`s_tir/dlight/c6678/softmax.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py) + [`tirx/transform/c6678_dma_lower.py::C6678AnnotateGlobalAlloc`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) | ✅（1925 chars 出码 + 7/7 形式化回归 PASS） |

---

## 5. 还差什么没做

| 项 | 影响 | 入口建议 |
|---|---|---|
| **A.10 step2**：`_p_` 内层退化开关 | 命中 `_p_` 形态时，把内层 32×32×32 三层 `for` 替换成 `call_extern("fp_matmul_fusion_p_", ...)`，便于复用旧字符串模板的 SIMD/intrinsic 实现；当前内层是 serial loop，没有 SIMD | 在 `MatmulGemmTemplate.apply` 内层 reorder 之后，看 staging stride 是否与 BSP `_p_` ABI 紧凑步长一致；若一致追加 `sch.annotate(... "c6678.p_inner")`，并在 A.9 后挂一段 `_p_` rewrite pass |
| **A.6 完成度**：当前已支持 `Matmul` + `Softmax`（PR-S2 phase A）+ `ElementGreaterEqual` same-shape；conv / reduce / layernorm / RMSNorm / LSTM pattern 仍未完成 | ElementGreaterEqual 已能生成 `int8_t*` bool 输出、多核派发、输入侧 `dma_trans` staging 和 1D L2 compact，但 broadcast、scalar 分支、输出 DMA store、BSP `get_l2_addr` 指针化和 `fp_greater_equal_s` extern 退化还未接；LSTM 目前只有 extern wrapper，不是自动识别 | 新算子继续按 `softmax.py` / `elementwise.py` 模式新增 `ScheduleRule`；ElementGreaterEqual 下一步补 broadcast 特征识别、输出 DMA store 与 BSP `get_l2_addr` 指针化；LSTM 下一步需要先定义高层 pattern 或 Relax/TIR 复合算子边界 |
| **`dma_trans` 自动调度尚未用于 softmax phase B** | A.3/A.9 已支持 `dma_trans` 合法性校验和 1D annotation lowering；ElementGreaterEqual 已开始使用输入侧 `dma_trans` staging，但 softmax phase A 仍是栈数组 + 串行内层 | 下一步把 ElementGreaterEqual 的 1D compact 机制迁移到 softmax phase B 的分阶段 reduction，并补输出 DMA store |
| **storage plan 未被消费** | A.2 只写 attrs；L2 buffer 的 `region_base/end` 仍占位 `-1`，最终源码不感知 storage plan | 等 `C6678MulticoreLower` 拿到 `core_id` 后回填，再让 codegen 端打印对应 L2 地址 |

---

## 6. 想新增一个算子（如 conv / layernorm）：从哪一步开局

> **PR-S2 phase A 已落地参考样例**：[`dlight.c6678.Softmax`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py) + 端到端 demo [`generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) + 形式化回归 [`test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py)，可作为新算子开局模板。详细技术发现见 [`learning.md` §4.9](./learning.md#49-pr-s2-phase-a-关键发现2026-05-25)。

参考 [`examples.md`](./examples.md) 提供的 BSP 端 C 实现（覆盖纯 L2 + DMA 两种路径），新增一个 dlight 模板的最小工作量：

1. **PrimFunc 写法注意**（PR-S2 phase A 教训，必读）：
   * 中间 buffer **必须**用 `T.sblock_alloc_buffer((shape,), "dtype")`，**不能**用 `T.alloc_buffer`。前者把 buffer 加到 root SBlock 的 `alloc_buffers`，后者生成独立 `AllocBuffer` 语句节点 —— 后者会让 `compute_at` 的 `IsOutputBlock` 检查把中间 buffer 误判为 output（参见 [`learning.md` §4.9.1](./learning.md#491-primfunc-中间-buffer-必须用-tsblock_alloc_buffer不能用-talloc_buffer)）。
   * `T.exp` / `T.log` / `T.sqrt` 等 intrin 默认会通过 [`tvm/target/intrin.py`](file:///home/tangqingyun/tvm/python/tvm/target/intrin.py) 的 fp32 默认规则降为 `expf(...)` / `logf(...)` / `sqrtf(...)`（C99 `<math.h>`），**不需要** c6678 专属 intrin。BSP 端只要保证链入 C99 标准库即可。
2. **抽特征**（A.4）已经能识别 `softmax` 的 `op_kind`；新算子若不能识别，先在 [`c6678_features.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/analysis/c6678_features.py) 加一个 `_classify_xxx_block` 分支返回对应 `op_kind`。
3. **加模板**：在 `s_tir/dlight/c6678/` 新增 `conv.py`（或 `layernorm.py` 等），写一个 `XxxRule(C6678ScheduleRule)`：
   * `apply()` 里取出 PrimFunc，用 `normalize_prim_func(sch)` 拿 `block_infos`，识别该算子的特征 4/N 块结构（参考 [`softmax.py::_identify_softmax_blocks`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py#L125)）；
   * 调度顺序 **必须**与 `dlight/cpu/reduction.py` 一致：先 `sch.parallel(outer)`，再 `for bi in reversed(block_infos[:-1]): sch.compute_at(bi.block_rv, outer, preserve_unit_loops=True)`；
   * 后续若需要 staging（PR-S2 phase B），按 `block_capacity`（fp32 = 115200，i8 = 32768，参考 [`examples.md`](./examples.md)）做 `cache_read("global.l2")` + `sch.annotate("c6678.dma_load", "dma_trans")`，让 [`C6678DMALower`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py) 接管出码。
4. **登记到 dlight 入口**：在 [`s_tir/dlight/c6678/__init__.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/__init__.py) 中追加 `from .xxx import Xxx`，并在 `dlight.ApplyDefaultSchedule(c6678_dlight.Xxx())(mod)` 处使用。
5. **不需要操心 large alloca**：[`C6678AnnotateGlobalAlloc`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) 已挂在主 pipeline，自动给所有 `scope='global'` 的 `AllocBuffer` 加 `disable_lower_builtin=True` 注解，绕过 c6678 bare-metal 没有 `device_id` 时 `LowerTVMBuiltin` 走 `TVMBackendAllocWorkspace` 的崩溃路径（参见 [`learning.md` §4.9.2](./learning.md#492-c6678-上-scopeglobal-大-alloc-必须打-disable_lower_builtin--true)）。
6. **加冒烟测试**：仿 [`test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py) 的 7 用例结构，写一个新算子 TVMScript 输入，断言出码里出现：
   * `void xxx_xxx(float*, float*, ..., int32_t core_mask)`；
   * 多核派发 `if ((GetCoreNum > 0) && (c6678_get_core_id >= 0)) { for ... }`；
   * `C6678E_SyncN(...)` 收尾；
   * 中间 buffer 的栈数组（如 `T_xxx[N]`），且不出现 `TVMBackendAllocWorkspace`；
   * 出码字节数零差分快照。

最终对用户来说还是一行：

```python
runtime_mod = tvm.tirx.build(xxx_mod, target=target)
```

只是 `dlight.ApplyDefaultSchedule` 那一步会自动选到 `Xxx` rule 而不是 `Matmul` / `Softmax` 而已。

---

## 附：关键路径速查

* 主路线图：[`learning.md`](./learning.md)
* 历史 / SOP / 命令白名单：[`dev_history.md`](./dev_history.md)
* API 参考：[`api_reference.md`](./api_reference.md)
* BSP 实现样例：[`examples.md`](./examples.md)
* 流水线源代码：[`pipeline.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py)
* c6678 dlight 入口：[`s_tir/dlight/c6678/__init__.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/__init__.py)
* c6678 transform 集合：[`tirx/transform/`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform)


## 8. LSTM 的当前定位

`examples.md` 中的 LSTM 是 BSP 侧复合算子，实现里包含 gate matmul、pack、sigmoid/tanh、state update、zoneout、projection、双向分支和多核同步。当前 TVM 侧已新增 `test_c6678_lstm_extern_codegen.py`，可以生成一个 `lstm_fp32_extern(Output, Input, Params, core_mask)` wrapper，并在 C 源码中调用 `fp_lstm_s(Output, Input, Params, core_mask)`。

这意味着 LSTM 已经可以作为“外部复合 kernel”挂到 c6678 出码链路里，但还没有完成从普通 TVMScript/Relax LSTM 子图自动识别并拆成专家模板的能力。中期展示时建议明确区分：`LSTM extern wrapper 已接入`，`LSTM 自动模板化仍是下一阶段工作`。


## 9. L2/DMA 分块当前状态

当前已经完成 ElementGreaterEqual 的输入侧 DMA staging + 1D L2 compact：schedule 为 A/B 两个输入插入 `cache_read("global.l2")`，并通过 `c6678.dma_load="dma_trans"` 触发 `C6678DMALower`。该 pass 会生成带 tile offset 的 `dma_trans`，修正 tail tile size，并把 L2 buffer 访问改写成 tile-local index。当前 `N=262144` 回归中，A/B 两个 87040 元素 tile 会被 StorageRewrite 合并成 `float A_global_l2[174080]`，不再分配全量 `524288`。

尚未完成的是更贴近 `examples.md` 的 BSP pointer-style L2 region planning：当前源码仍是 C 栈数组形式的 compact L2 storage，而不是显式 `get_l2_addr(core_id)` + 手动分区指针。输出 DMA store、broadcast/scalar GreaterEqual、softmax phase B 也仍待继续。
