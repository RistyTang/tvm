# C6678 DSP API 参考

> 本文是 [`learning.md`](./learning.md) 的附录，用于让接手 agent 快速找到"目前能调用什么、参数怎么填、返回什么"。
>
> 所有内容与代码一致：
> - `python/tvm/contrib/c6678.py`
> - `python/tvm/tirx/c6678_config.py`
> - `python/tvm/tirx/transform/c6678_storage_plan.py`
> - `src/target/target_kind.cc` 中的 c6678 TargetKind

目录：

1. c6678 Target 属性表
2. `tvm.tirx.c6678_config`：硬件常量唯一事实源
3. `tvm.tirx.transform.C6678StoragePlan`：storage planning pass
4. `tvm.contrib.c6678`：matmul MVP 入口
5. 调用关系总览
6. `tvm.s_tir.dlight.c6678`：c6678 schedule 规则（A.5/A.6 起步）
7. `tvm.s_tir.dlight.c6678.ElementGreaterEqual`：same-shape elementwise rule
8. LSTM extern composite wrapper
9. `tvm.tirx.analysis.C6678Features` 与 L2 门禁
10. `C6678DMALower` 失败策略

---

## 1. c6678 Target 属性表

`Target("c6678")` 当前可识别的硬件属性如下，全部由 `src/target/target_kind.cc` 注册并写入了与板卡一致的 `DefaultValue`。

| attr | 默认值 | 含义 |
|---|---|---|
| `core_num` | `8` | 6678 核心总数 |
| `core_freq_mhz` | `1250` | 核频率 1.25 GHz |
| `l1_size` | `32 KiB` | L1 大小，仅占位（当前不参与 DMA） |
| `l2_size` | `0x000F0000` | 每核可用 L2 窗口大小（对应物理地址 `0x00810000 ~ 0x008FFFFF`） |
| `l2_base_core0` | `0x10810000` | core 0 的可用 L2 起始地址 |
| `l2_core_stride` | `0x01000000` | 相邻核心 L2 之间的步长 |
| `smc_base` | `0x0C000000` | 全核共享 SMC 起始地址 |
| `smc_size` | `0x00800000` | SMC 总大小 8 MB |
| `ddr_base` | `0x80000000` | DDR 起始地址 |
| `ddr_size` | `0x80000000` | DDR 默认窗口（开发假设） |
| `dma_align_bytes` | `64` | DMA 推荐对齐 |
| `dma_burst_bytes` | `64` | DMA burst 单位 |
| `dma_max_transfer` | `0x7FFFFFFF` | 单次最大搬运字节数（INT_MAX） |
| `vector_bytes` | `32` | 向量寄存器宽度（开发假设） |
| `workspace-byte-alignment` | 未填 | TVM 公共字段 |
| `constants-byte-alignment` | 未填 | TVM 公共字段 |

### 重要约束

- `Target("c6678").attrs` 在用户未显式指定时，**Python 端不会自动带上 C++ 的 DefaultValue**。
  所以下游 pass 必须通过 `tvm.tirx.c6678_config.from_target(target)` 取值，
  让 `_C6678_DEFAULT_ATTRS` 兜底；详见 §2。
- 用户可以通过 `Target("c6678 -l2_size=...")` 覆盖默认值。

---

## 2. `tvm.tirx.c6678_config`

> 文件：[`python/tvm/tirx/c6678_config.py`](../python/tvm/tirx/c6678_config.py)

后续所有 c6678 专属 pass / schedule / template 的"硬件常量唯一事实源"。

### 2.1 顶层符号

| 符号 | 类型 | 含义 |
|---|---|---|
| `C6678_MAX_CORES` | `int = 8` | 核心总数常量 |
| `_C6678_DEFAULT_ATTRS` | `dict[str, int]` | 14 项硬件参数默认值表（与 `target_kind.cc` 严格对齐） |
| `C6678Config` | `@dataclass(frozen=True)` | 只读硬件配置快照 |
| `from_target(target)` | 函数 | 根据 `target` 构造 `C6678Config` |

### 2.2 `C6678Config`

只读 dataclass，14 个字段一一对应 §1 中的 attr：

```text
core_num, core_freq_mhz, l1_size, l2_size,
l2_base_core0, l2_core_stride,
smc_base, smc_size, ddr_base, ddr_size,
dma_align_bytes, dma_burst_bytes, dma_max_transfer, vector_bytes
```

#### 方法

| 方法 | 签名 | 行为 |
|---|---|---|
| `l2_address_range` | `(core_id: int) -> tuple[int, int]` | 返回核心 `core_id` 的 L2 物理地址区间 `(base, end_inclusive)`；`core_id` 越界抛 `ValueError`。 |
| `smc_address_range` | `() -> tuple[int, int]` | 返回 SMC 区间。 |
| `ddr_address_range` | `() -> tuple[int, int]` | 返回 DDR 区间。 |
| `iter_cores_from_mask` | `(core_mask: int) -> list[int]` | 把 `core_mask` 位图展开成参与核心 id 列表；`<= 0` 或 `>> core_num` 时抛 `ValueError`。 |

### 2.3 `from_target`

```python
def from_target(target: str | Target) -> C6678Config
```

- `target.kind.name` 必须是 `"c6678"`，否则抛 `ValueError`。
- 内部对每个属性走 `_read_attr`：先读 `target.attrs`，缺失时回退 `_C6678_DEFAULT_ATTRS`。
- **下游 pass 不得直接读 `target.attrs`，必须通过本函数获取硬件参数**。

#### 调用示例

```python
from tvm.target import Target
from tvm.tirx import c6678_config

cfg = c6678_config.from_target(Target("c6678"))
print(cfg.l2_address_range(3))      # (0x13810000, 0x138FFFFF)
print(cfg.iter_cores_from_mask(0xF)) # [0, 1, 2, 3]
```

---

## 3. `tvm.tirx.transform.C6678StoragePlan`

> 文件：[`python/tvm/tirx/transform/c6678_storage_plan.py`](../python/tvm/tirx/transform/c6678_storage_plan.py)

C6678 专属 storage planning pass（**识别但不改写** 形态）。

### 3.1 触发条件

仅当 `func.attrs["target"].kind.name == "c6678"` 才生效；否则原样返回 `func`，可以安全挂到任何 pipeline。

### 3.2 构造与签名

```python
@prim_func_pass(opt_level=0, name="C6678StoragePlan")
class C6678StoragePlan:
    def __init__(self, strict: bool = False) -> None: ...
    def transform_function(self, func, mod, ctx) -> PrimFunc: ...
```

| 参数 | 类型 | 默认 | 含义 |
|---|---|---|---|
| `strict` | `bool` | `False` | True 时遇到非 `l2/smc/ddr/global` 的 buffer scope 直接抛 `ValueError`；False 时记录到结果供下游决定。 |

### 3.3 行为

1. 遍历 `func.buffer_map` 中所有外部 buffer。
2. 把 `buf.scope()` 通过 `_normalize_scope` 抹平为 `l2/smc/ddr/global` 之一
   （兼容 `""`、`"global"`、`"global.l2"`、`"l2"` 等形态）。
3. 为每个 buffer 生成一条 plan entry，写回 `func.attrs["c6678.storage_plan"]`。
4. **不改 IR**，仅 `func.with_attr(...)`。

### 3.4 输出 schema：`c6678.storage_plan`

`func.attrs["c6678.storage_plan"]` 是一个 list，每个 entry 字段：

| 字段 | 类型 | 含义 |
|---|---|---|
| `param` | `str` | `buffer_map` 中对应的形参名 |
| `buffer` | `str` | `buf.name` |
| `scope` | `str` | 归一化后 `l2 / smc / ddr / global` 之一 |
| `region_base` | `int` | 物理地址下界；`l2` 暂填 `-1`（待 multicore lowering 拿到 `core_id` 再回填） |
| `region_end` | `int` | 物理地址上界（`end_inclusive`）；`l2` 暂填 `-1` |

> `scope == "global"` 在本 pass 中按 DDR 处理（即用 `cfg.ddr_address_range()`）。
> 若未来引入"显式 unstaged"概念，再独立区分。

### 3.5 调用示例

```python
import tvm
from tvm.script import tirx as T

@T.prim_func
def f(A: T.Buffer((16, 16), "float32"), ...):
    T.func_attr({"target": T.target("c6678"), "global_symbol": "f"})
    ...

mod = tvm.IRModule({"f": f})
mod = tvm.tirx.transform.C6678StoragePlan()(mod)
plan = mod["f"].attrs["c6678.storage_plan"]
```

> 当前 pass 已挂入 `s_tir.pipeline.default_s_tir_pipeline` 的 c6678 专属前置阶段；也可以手动调用用于单 pass 验证。注意：该 pass 仍只写 `c6678.storage_plan` attrs，下游暂未消费这些 region 信息。

---

## 4. `tvm.contrib.c6678`

> 文件：[`python/tvm/contrib/c6678.py`](../python/tvm/contrib/c6678.py)

C6678 matmul MVP 的 Python 入口，基于参数 + target attrs 直接生成 C source module。

### 4.1 顶层入口

#### `build_matmul_module(...)` → `runtime.Module`

签名（关键字参数）：

```python
def build_matmul_module(
    *,
    M: int,
    N: int,
    K: int,
    dtype: str = "float32",
    target: str | Target = "c6678",
    transpose_a: bool = False,
    transpose_b: bool = False,
    activation: str = "none",          # none / relu / relu6
    bias_broadcast: bool = True,
    a_scope: str = "ddr",              # ddr / l2 / smc
    b_scope: str = "ddr",
    c_scope: str = "ddr",
    use_multicore: bool = True,
    core_mask: int = 0xFF,
    block_size: int | None = None,     # None 时按 dtype 选默认
    # 以下为可选硬件覆盖参数；None 时优先读 target.attrs，再回退默认
    core_num: int | None = None,
    l2_size: int | None = None,
    l2_base_core0: int | None = None,
    l2_core_stride: int | None = None,
    smc_base: int | None = None,
    smc_size: int | None = None,
    ddr_base: int | None = None,
    ddr_size: int | None = None,
    dma_align_bytes: int | None = None,
    dma_max_transfer: int | None = None,
    vector_bytes: int | None = None,
) -> tvm.runtime.Module
```

构建前会做三件事：

1. dtype 必须落在 `_SUPPORTED_DTYPES`；`a/b/c_scope` 必须落在 `l2/smc/ddr`，否则抛错。
2. `core_mask` 必须是 `[1, 0xFF]` 内的有效位图；`use_multicore=False` 时强制为 `0x01`。
3. 硬件常量先读 `target.attrs`，再读用户显式参数，最后回退默认。

返回的是 `tvm.runtime._ffi_api.CSourceModuleCreate(source, "c6678", function_names, None)`，
导出函数名形如 `${prefix}_matmul_fusion_${variant}`，其中：

- `prefix ∈ {fp, dp, i8, i16, i32}` 由 dtype 决定（见 `_SUPPORTED_DTYPES`）。
- `variant`：`a/b/c_scope` 全为 `l2` 时为 `p`（pure on-chip），否则为 `s`（含 DMA）。

#### `build_matmul_source(**kwargs) -> str`

等价于 `build_matmul_module(**kwargs).inspect_source("c6678")`，
用于不落盘的快速肉眼审查。

### 4.2 工具函数

| 函数 | 签名 | 行为 |
|---|---|---|
| `get_l2_address_range` | `(core_id, *, l2_base_core0, l2_core_stride, l2_size) -> (int, int)` | 返回核 `core_id` 的 L2 区间。默认参数取 `_C6678_DEFAULT_ATTRS`。 |
| `iter_cores_from_mask` | `(core_mask: int) -> list[int]` | 8-bit 位图展开成核心 id 列表。 |
| `validate_dma_path` | `(src_scope: str, dst_scope: str) -> None` | 校验 DMA 端点是否落在 `l2/smc/ddr`，否则抛 `ValueError`。仅用于字符串模板旧路径，不是 `load_row_major_tile` 新 ABI 的事实源。 |

### 4.3 常量与映射表

| 名称 | 来源 | 用途 |
|---|---|---|
| `_SUPPORTED_DTYPES` | 模块内 | dtype → `{prefix, c_type, acc_type, block_size, activation_zero, activation_six}` |
| `_SCOPE_TO_ENUM` | `{"ddr": 0, "l2": 1, "smc": 2}` | 生成 C 源码 enum |
| `_ACTIVATION_TO_ENUM` | `{"none": 0, "relu": 1, "relu6": 2}` | 生成 C 源码 enum |
| `C6678_MAX_CORES` | re-export from `tirx.c6678_config` | 8 |
| `C6678_DEFAULT_L2_BASE_CORE0` 等 | re-export from `_C6678_DEFAULT_ATTRS` | 兼容旧 import 路径 |

> 所有常量已迁移到 `tvm.tirx.c6678_config`，`contrib/c6678.py` 仅做 re-export。
> **新代码请优先 import `tvm.tirx.c6678_config`，不要再扩展 contrib 一侧的常量。**

### 4.4 调用示例

```python
from tvm.contrib import c6678

mod = c6678.build_matmul_module(
    M=128, N=128, K=128,
    dtype="float32",
    a_scope="ddr", b_scope="ddr", c_scope="ddr",
    use_multicore=True, core_mask=0xFF,
)
print(mod.inspect_source("c6678"))
```

更多示例见 [`generate_c6678_matmul.py`](./generate_c6678_matmul.py)
与 [`tests/test_c6678_matmul_codegen.py`](./tests/test_c6678_matmul_codegen.py)。

---

## 5. 调用关系总览

```
┌─────────────────────────────────────────────────────────┐
│  src/target/target_kind.cc       (c6678 TargetKind)     │
│  ─ 注册 14 项 hardware attrs + DefaultValue             │
└──────────────────────┬──────────────────────────────────┘
                       │ Target("c6678").attrs
                       ▼
┌─────────────────────────────────────────────────────────┐
│  tvm.tirx.c6678_config           (Python 单一事实源)    │
│  ─ _C6678_DEFAULT_ATTRS  (Python 端兜底)                │
│  ─ C6678Config  +  from_target()                        │
└─────────┬──────────────────────────────────┬────────────┘
          │                                  │
          ▼                                  ▼
┌──────────────────────────┐    ┌────────────────────────┐
│ tvm.tirx.transform        │    │ tvm.contrib.c6678      │
│ .C6678StoragePlan         │    │ .build_matmul_module   │
│ ─ 写 c6678.storage_plan   │    │ ─ 直接生成 C source     │
└──────────────────────────┘    └────────────────────────┘
          │                                  │
          ▼                                  ▼
   下游 pass / schedule              板卡侧 / 部署链路
   （A.3 DMALegalize、
    A.4 特征提取、
    A.5 模板派发、
    A.6 matmul schedule）
```

接手时只需记住一条规则：**任何需要硬件常量的新代码，先 `from tvm.tirx.c6678_config import from_target`，再从 `C6678Config` 取值。**

---

## 6. `tvm.s_tir.dlight.c6678`

c6678 专属的 dlight schedule 规则集合，按 `s_tir/dlight/cpu`、`s_tir/dlight/gpu` 同构组织，
通过 `ApplyDefaultSchedule` 在 `tvm.tirx.build` 之前对 PrimFunc 套用 schedule。

### 6.1 顶层符号

```python
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight

dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)
```

| 符号 | 类型 | 含义 |
|---|---|---|
| `tvm.s_tir.dlight.c6678.Matmul` | `ScheduleRule` 子类 | matmul schedule（dispatcher + GEMM template + L2/DMA staging） |
| `tvm.s_tir.dlight.c6678.Softmax` | `ScheduleRule` 子类 | softmax phase A schedule（4 块 max/exp/sum/div + outer parallel） |
| `tvm.s_tir.dlight.c6678.ElementGreaterEqual` | `ScheduleRule` 子类 | same-shape `float32 >= float32 -> bool` elementwise MVP |
| `tvm.s_tir.dlight.c6678.base.C6678ScheduleRule` | `ScheduleRule` 子类 | c6678 规则的公共基类，仅在 `target.kind.name == "c6678"` 时生效 |

### 6.2 `Matmul.apply` 行为约束

仅当满足以下条件时返回 schedule，其它情况返回 `None`：

1. `func` 是 `tirx.PrimFunc` 且 `target.kind.name == "c6678"`；
2. `normalize_prim_func` + `try_inline_contiguous_spatial` 后只剩**一个** block；
3. block 的 `dom_kind() == "SSR"` 且循环嵌套深度恰好为 3；
4. 三个循环 extent 都是常量（IntImm）。

满足条件后的 schedule 等价于：

```python
i_outer, i_inner = sch.split(i, [None, ti])
j_outer, j_inner = sch.split(j, [None, tj])
k_outer, k_inner = sch.split(k, [None, tk])
sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)
sch.parallel(i_outer)
sch.annotate(k_inner, "pragma_auto_unroll_max_step", tk)
sch.annotate(k_inner, "pragma_unroll_explicit", 1)
```

`ti / tj / tk` 由 `_pick_factor(extent, (32, 16, 8, 4, 2, 1))` 选出能整除对应 extent
的最大候选，否则退化为 `1`（即不切分）。

### 6.3 已知边界

- 当前 codegen `codegen_c6678.cc::VisitStmt_(ForNode*)` 将所有循环按 serial 输出，`sch.parallel`
  暂时只是占位，给未来 A.5 多核 lower 留接口。
- 未做 `cache_read` / `cache_write` / `dma_copy` / `tensorize`，性能仍接近 baseline，
  以"端到端跑通"为本期目标。
- `Matmul` 只匹配 `[S,S,R]` 形态；`[B,S,S,R]`（带 batch）和带 epilogue 的 fusion
  会落到 A.5 通用 dispatcher 时再处理。

### 6.4 端到端用法

完整示例见 [`generate_c6678_matmul_via_build.py`](./generate_c6678_matmul_via_build.py)：

```python
import tvm
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T

@T.prim_func
def matmul_fp32(A: T.Buffer((128, 128), "float32"),
                B: T.Buffer((128, 128), "float32"),
                C: T.Buffer((128, 128), "float32")):
    T.func_attr({"global_symbol": "matmul_fp32", "tir.noalias": True})
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]

mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
target = tvm.target.Target("c6678")
with target:
    mod = dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)
runtime_mod = tvm.tirx.build(mod, target=target)
print(runtime_mod.inspect_source("c6678")[:2000])
```



## 7. `tvm.s_tir.dlight.c6678.ElementGreaterEqual`

> 文件：[`python/tvm/s_tir/dlight/c6678/elementwise.py`](../python/tvm/s_tir/dlight/c6678/elementwise.py)

当前 rule 是 ElementGreaterEqual 的输入 DMA staging MVP：

| 项 | 当前契约 |
|---|---|
| 输入 PrimFunc | 单个无 reduction 的 injective block |
| 匹配表达式 | block body 是 `BufferStore`，store value 是 `tirx.GE` |
| 调度动作 | `split(outer, block_elems)`、两个输入 `cache_read("global.l2")`、`compute_at(tile_outer)`、`c6678.dma_load="dma_trans"`、`parallel(tile_outer)` |
| 多核派发 | 由后续 `C6678MulticoreLower` 自动加入 `core_mask`、`GetCoreNum`、`GetLogicCoreId(core_mask, DNUM)`、`C6678E_SyncN` |
| 输出 dtype | TVM bool storage 在 C codegen 中落为 `int8_t*` |
| 已完成 | same-shape 输入侧 `dma_trans` staging + 1D L2 compact，tile offset、tail size、tile-local L2 index 由 `C6678DMALower` 计算 |
| 未完成 | scalar/broadcast 分支、输出 DMA store、BSP `get_l2_addr` 指针化、直接退化到 BSP `fp_greater_equal_s` |

验证入口：[`Test4dsp/tests/test_c6678_greater_equal_codegen.py`](./tests/test_c6678_greater_equal_codegen.py)。


### 7.1 L2 DMA block helper

> 文件：[`python/tvm/tirx/c6678_config.py`](../python/tvm/tirx/c6678_config.py)

`l2_dma_block_elems(dtype, l2_size, num_input_buffers, num_output_buffers, reserve_bytes=4096, align_bytes=64, max_elems=None)` 使用等分 L2 的保守模型计算 1D tile 容量。当前 ElementGreaterEqual 按 2 个输入 staging + 1 个预留输出 staging 估算，即使输出暂未 DMA store，也为后续扩展保留 L2 空间。

### 7.2 1D dma_trans compact

当前 ElementGreaterEqual 的 1D 输入 staging 已在 `C6678DMALower` 内完成专用 compact：pass 在 `ConvertBlocksToOpaque` 前消费 `SBlockRealize.iter_values`，记录 tile 起点和 tile extent，生成 `dma_trans((&A[offset]), &L2[0], size_bytes)`，并把后续 L2 `BufferLoad/BufferStore` 从全局 index 改写为 `index - tile_start`。这样下游 `StorageRewrite` 可把 A/B 两个 compact tile 合并为 `2 * tile_extent` 的 L2 storage，避免原始全量 shape 分配。

该逻辑只覆盖 `c6678.dma_load="dma_trans"` 的 1D 连续 staging，不影响 matmul 的 `load_row_major_tile` 2D 路径。本轮源码契约已把 `global.l2` alloc 的目标形态切到 per-core L2 指针，即 `l2_base_core0 + DNUM * l2_core_stride`；仓库中已提交的生成快照若仍是旧栈数组，需要重新运行生成脚本刷新。

## 8. LSTM extern composite wrapper

> 验证入口：[`Test4dsp/tests/test_c6678_lstm_extern_codegen.py`](./tests/test_c6678_lstm_extern_codegen.py)

当前 LSTM 不是 dlight pattern rule，而是 extern composite kernel wrapper。TVMScript wrapper 形态为：

```python
T.evaluate(T.call_extern("", "fp_lstm_s", Output.data, Input.data, Params.data, core_mask))
```

生成 C 源码中会保留 bare-C 入口，并调用 BSP 侧 `fp_lstm_s(Output, Input, Params, core_mask)`。这一步用于证明 LSTM 可以纳入 c6678 出码链路；自动识别 LSTM 子图、拆分 gate matmul/sigmoid/tanh/state update 并生成专家 schedule 仍未实现。

## 9. `tvm.tirx.analysis.C6678Features` 与 L2 门禁

> 文件：[`python/tvm/tirx/analysis/c6678_features.py`](../python/tvm/tirx/analysis/c6678_features.py)

`extract_features(func, target)` 是 A.5 dispatcher 的只读输入契约。当前输出 `C6678PrimFuncFeatures`，每个 block 给出：`op_kind`、`dom_kind`、`dom_extents`、`dtype`、`read_bufs`、`write_bufs`、`flop_count_static`、`tile_hint`、`static_alloc_l2_bytes`。

当前 L2 容量估算采用保守口径：对 matmul 的读 buffer staging 按 A/B 两个原始读 buffer 的静态 shape 全量估算，而不是只估算单个 tile。128×128×128 fp32 matmul 的 `static_alloc_l2_bytes` 为 131072B。该值供 `dispatcher.select_template` 与 `C6678Config.l2_size` 做容量门禁，不应再被解释为“最终一定生成 `float A_global_l2[32768]` 栈数组”。

## 10. `C6678DMALower` 失败策略

> 文件：[`python/tvm/tirx/transform/c6678_dma_lower.py`](../python/tvm/tirx/transform/c6678_dma_lower.py)

`C6678DMALower` 支持两类 annotation：

| annotation | 期望形态 | 输出 |
|---|---|---|
| `c6678.dma_load="load_row_major_tile"` | 严格 2 层 For 包裹 staging `SBlockRealize` | `call_extern("load_row_major_tile", src, dst, row0, col0, rows, cols, src_ld, elem_bytes)` |
| `c6678.dma_load="dma_trans"` | 严格 1 层 For 包裹 staging `SBlockRealize` | `call_extern("dma_trans", src, compact_dst, size_bytes)`，并同步做 1D L2 compact remap |

如果 pass 执行后仍残留 `c6678.dma_load` annotation，说明 schedule 端生成了无法识别的 DMA staging 形态，当前实现会直接抛 `ValueError`，避免静默跳过后仍标记 `c6678.dma_lowered=True`。
