# C6678 DSP 开发历史与 SOP 附录

> 本文是 [`learning.md`](./learning.md) 的附录，承载历史、流程性内容；主文档只保留架构与路线图。
>
> **何时来这里**：想知道某条规则当时怎么进来的、想确认开发流程、想找到完整的命令白名单与示例。

## 1. 命令白名单（开发环境）

**所有 Python 类命令必须运行在 `conda tvm_env` 环境中**。涉及 `python xxx.py` / `pytest` / 任何依赖 `tvm`/`tvm_ffi`/`numpy` 的脚本，都必须先：

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh
conda activate tvm_env
```

否则容易出现 `tvm_ffi` 导入异常 / `core` 循环导入 / 找不到 `tvm.contrib` / Python 与本地编译产物不匹配。

### 1.1 推荐的 PYTHONPATH

```bash
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python
```

完整模板：

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python your_script.py
```

### 1.2 已验证的命令模板

#### A. 运行 Python 脚本

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python /home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul.py
```

#### B. 运行内联 Python 验证

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python - <<'PY'
from tvm.contrib import c6678
mod = c6678.build_matmul_module(M=32, N=32, K=32, dtype='float32', target='c6678')
print(mod.inspect_source('c6678')[:200])
PY
```

#### C. 直接调用测试函数（绕开 pytest plugin 初始化问题）

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python - <<'PY'
import importlib.util
spec = importlib.util.spec_from_file_location(
    'test_c6678_matmul_codegen',
    '/home/tangqingyun/tvm/Test4dsp/tests/test_c6678_matmul_codegen.py',
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
mod.test_generate_c6678_matmul_source_for_ddr()
mod.test_generate_c6678_matmul_source_for_l2()
print('ok')
PY
```

#### D. 验证 A.2 storage plan pass

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh
conda activate tvm_env
python /home/tangqingyun/tvm/Test4dsp/tests/test_c6678_storage_plan.py
```

### 1.3 不推荐的方式

1. **不要在未激活 `tvm_env` 的环境中运行 Python**。
2. **不要默认依赖 `pytest` 直接跑通所有用例**：当前 TVM testing plugin 初始化时会检查 `llvm/cuda/opencl/...` 等可用 target；本地 TVM 构建未必启用全部，会在 plugin 阶段就报错。优先用方案 C 直接调用测试函数。


### 1.4 中期展示前推荐验证命令

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && conda activate tvm_env && python Test4dsp/tests/test_c6678_features.py && python Test4dsp/tests/test_c6678_dispatcher.py && python Test4dsp/tests/test_c6678_dma_lower.py && python Test4dsp/tests/test_c6678_end_to_end.py && python Test4dsp/tests/test_c6678_softmax_codegen.py && python Test4dsp/tests/test_c6678_greater_equal_codegen.py && python Test4dsp/tests/test_c6678_lstm_extern_codegen.py && python Test4dsp/tests/test_my_ops.py
```

历史上已验证：features/dispatcher/DMA lower failure/matmul 端到端/softmax/ElementGreaterEqual/LSTM extern/FFI demo 均通过。输出中可能出现 LLVM canonicalizer warning，原因是本地 TVM 未启用 LLVM target，和 c6678 路径本身无关。

### 1.5 本轮真实阻塞与闭环（2026-06-03）

- 本轮已完成源码修改：`c6678_multicore_lower.py`、`c6678_dma_lower.py`、`c6678_dma_legalize.py`、`codegen_c6678.cc/.h`、`c6678_config.py`、`target_kind.cc` 以及相关测试已改到新 ABI 方向。
- 目标 ABI 为：
  - `GetLogicCoreId(core_mask, DNUM)`
  - `load_row_major_tile(void* src_base, void* dst, int row0, int col0, int rows, int cols, int src_ld, int elem_size)`
  - `float* A_global_l2 = (float*)(l2_base_core0 + DNUM * l2_core_stride)`
- 中途确实遇到过终端执行工具异常：尝试运行 `python Test4dsp/generate_c6678_matmul_via_build.py` 与相关测试时，工具层一度返回 `unknown error: command 'icube.shellExec.runCommand' not found`。
- 随后重新编译 `libtvm_compiler.so` 并恢复终端调用后，本轮闭环已完成：
  1. `cmake --build . --parallel 4` 成功，`target_kind.cc` 与 `codegen_c6678.cc` 已重新链接进 `lib/libtvm_compiler.so`；
  2. [`generated_c6678_matmul_via_build.c`](file:///home/tangqingyun/tvm/Test4dsp/tests/generated_c6678_matmul_via_build.c) 已重新生成，实测包含：
     - `GetLogicCoreId(core_mask, DNUM)`
     - `float* A_global_l2 = (float*)(276889600 + DNUM * 16777216);`
     - 8 参数 `load_row_major_tile(...)`
  3. 直接执行测试脚本已通过：
     - [`test_c6678_end_to_end.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_end_to_end.py)
     - [`test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py)
     - [`test_c6678_greater_equal_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_greater_equal_codegen.py)
- `pytest` 入口仍会被 TVM testing plugin 的默认 target 探测卡住，因为当前构建未启用 `llvm`；这是测试框架环境问题，不是 c6678 lowering/codegen 回归。当前采用“直接执行脚本”的方式完成了功能验证。

---

## 2. 标准开发流程（SOP）

### SOP-0：先确认目标类型

开发前先判断本次任务属于哪类：

1. target 能力补充
2. 源码生成器增强
3. 新算子模板接入
4. 测试与验证
5. 编译链正式下沉

### SOP-1：先看现状，再定修改层级

| 需求 | 推荐改动层 |
|---|---|
| 快速验证某个规则 / 某个算子的 6678 代码结构 | `python/tvm/contrib/c6678.py` |
| 已确认要接入正式 lowering | `tirx / s_tir / src/target` |
| 已确认是稳定规则 | 不应长期只留在 `python/tvm/contrib/c6678.py` |

### SOP-1.5：新增规则时必须标记归宿

每新增一条规则，都必须明确记录：

1. 规则当前写在哪里
2. 规则最终应该写到哪里
3. 当前属于原型 / 过渡 / 正式实现

状态分级：

- `已下沉`：规则已写入 TVM 核心组件，不再依赖文档
- `已部分下沉`：已存在于 target/codegen 或辅助模块，但还没进 pass 层
- `已原型化`：仅在 `python/tvm/contrib/c6678.py` 这一过渡层
- `已确认`：文档中已确认，但尚未进入代码
- `未实现`：尚未实现

### SOP-2：先用单算子打样，再推动规则下沉

新增算子推荐顺序：

1. 增加 Python 入口
2. 增加 config 结构体
3. 增加模板实现
4. 增加示例脚本
5. 增加轻量测试
6. 识别可沉淀的规则
7. 将稳定规则下沉到 TVM 编译链

### SOP-3：新增算子的文件组织（以 `conv` 为例）

1. `python/tvm/contrib/c6678.py` 中新增入口（如 `build_conv_module`）
2. `Test4dsp/` 下新增生成示例（如 `generate_c6678_conv.py`）
3. `Test4dsp/tests/` 下新增轻量测试
4. 主 `learning.md` 中追加规则说明与版本选择策略

新增核心算子的优先顺序：`matmul` → `elementwise` → `gemv / conv / reduction`。

### SOP-4：每次开发都要同步记录

修改后至少补充以下信息到文档或变更说明：涉及文件、每个文件的核心修改、修改目的、影响范围、新增规则、后续扩展建议、该规则是否已写入 TVM 核心。

### SOP-5：优先轻量验证，而非默认 pytest

1. 先用内联 Python 检查能否成功生成 `CSourceModule`
2. 读取 `inspect_source('c6678')` 检查源码结构
3. 有测试函数就直接调用（见 §1.2 C）
4. 等环境更稳定后再接入标准 pytest

### SOP-6：导出函数命名规则

- 根据 `dtype` 决定前缀（`fp/dp/i8/i16/i32`）
- 根据 `scope` / 执行路径决定后缀（`_p` 全 L2 / `_s` 含 DMA）
- **只公开一个最终目标算子名**，其余辅助函数应当是 `static *_impl`

### SOP-6.5：判断"规则可否写进 TVM"的标准

满足以下条件时，应考虑从原型层迁到 TVM 核心：

1. 已被两个及以上场景复用
2. 不再频繁变动
3. 明确不是仅用于实验
4. 后续新算子会依赖该规则

典型例子：`DDR -> _s` / `全 L2 -> _p` / scope→storage 策略 / DMA legality。

### SOP-7：长期 ABI 收敛要求

最终生成代码风格目标：

```c
void funcname(void* self_handle, void* args, int32_t num_args, void* result)
```

MVP 期可保留简化接口；后续正式接入编译链时必须考虑 ABI 收敛。

---

## 3. 调用示例

### 3.1 DDR 路径 `float32` matmul

```python
from tvm.contrib import c6678

mod = c6678.build_matmul_module(
    M=256, N=256, K=256,
    dtype='float32',
    target='c6678',
    a_scope='ddr', b_scope='ddr', c_scope='ddr',
)
source = mod.inspect_source('c6678')
# 公开最终函数：fp_matmul_fusion_s
```

### 3.2 全 L2 路径 `float32` matmul

```python
from tvm.contrib import c6678

mod = c6678.build_matmul_module(
    M=256, N=256, K=256,
    dtype='float32',
    target='c6678',
    a_scope='l2', b_scope='l2', c_scope='l2',
    use_multicore=False,
)
source = mod.inspect_source('c6678')
# 公开最终函数：fp_matmul_fusion_p
```

### 3.3 验证 storage plan pass

```python
import tvm
from tvm.script import tirx as T

@T.prim_func
def f(A: T.Buffer((16, 16), "float32"),
      B: T.Buffer((16, 16), "float32"),
      C: T.Buffer((16, 16), "float32")):
    T.func_attr({"target": T.target("c6678"), "global_symbol": "f"})
    for i, j in T.grid(16, 16):
        with T.block("c"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]

mod = tvm.IRModule({"f": f})
mod2 = tvm.tirx.transform.C6678StoragePlan()(mod)
plan = mod2["f"].attrs["c6678.storage_plan"]
# plan 是一个列表，每项含 param/buffer/scope/region_base/region_end
```

---

## 4. 当前已知问题与注意事项

### 4.1 `Test4dsp/tests/test_my_ops.py` 已修复为可直接运行的 FFI demo

当前脚本会自动用本机 C 编译器把 `Test4dsp/src/total_test.c` 编译为 `Test4dsp/build/libtotal_test.so`，再通过 `tvm_ffi.load_module(str(so_path))` 加载模块并调用 `mod.add_one_c(x, y)`。旧的 `libinfo.load_lib_ctypes("libtotal_test.so")` 单参数调用已经移除。该测试用于验证“用户通过 TVM FFI 模块接口调用 C 算子”的体验，不代表 c6678 生成代码已经完成板上运行闭环。

### 4.2 pytest 因 TVM plugin 初始化失败

如果看到 `None of the following targets are supported by this build of TVM`，不是业务逻辑错误，而是测试环境初始化时缺少某些 target 支持。优先使用轻量验证方案（SOP-5）。

---

## 5. MVP 探索与设计变迁

### 5.1 `python/tvm/contrib/c6678.py` 的定位

它**不是**通用 compiler pass，而是一个 **目标平台专用源码生成器原型**：

1. 解析 Python 输入参数
2. 读取 `c6678 target attrs`
3. 根据 `dtype + scope` 决定目标最终算子版本
4. 生成一个 `c6678` 类型的 `CSourceModule`
5. 允许通过 `inspect_source('c6678')` 直接拿到源码文本

定位是**完整框架开发过程中的过渡层**，用于快速验证规则是否合理；最终规则会下沉到 `tirx / s_tir / codegen_c6678`。

### 5.2 当前生成源码中的关键结构

可能包含或使用：

- `C6678MatmulConfig`
- `dma_trans_2d`、`copy_2d_by_scope`
- `load_row_major_tile`、`load_transposed_tile`、`store_row_major_tile`
- `*_matmul_block_kernel`
- `*_matmul_fusion_p_impl` / `*_matmul_fusion_s_impl`
- 公开的 `*_matmul_fusion_p` 或 `*_matmul_fusion_s`

### 5.3 当前能力边界

1. 没有把通用 `tvm.build(target='c6678')` 自动识别 matmul 的全链路打通
2. 没有把 `DDR/L2/SMC` 变成完整通用的 IR pass
3. 没有真正的性能模型
4. 没有接真实板卡 runner
5. 没有自动搜索空间生成

但已经提供稳定起点：可以根据规则生成正确命名的最终目标算子；可以沉淀对其他算子的模板式开发方法。

### 5.4 当前算子选择规则（已实现，必须遵守）

dtype → 前缀：

| dtype | 前缀 |
|---|---|
| `float32` | `fp` |
| `float64` | `dp` |
| `int8` | `i8` |
| `int16` | `i16` |
| `int32` | `i32` |

scope → 最终算子版本：

| 条件 | 输出函数 |
|---|---|
| `A/B/C` 全在 `L2` | `${prefix}_matmul_fusion_p` |
| 其余情况（含 `DDR`） | `${prefix}_matmul_fusion_s` |

- `_p`：数据已在片上，不需要 DDR→L2 的 DMA 搬运
- `_s`：包含 DMA 搬运、分块、多核等更复杂路径

### 5.5 第一批 elementwise 算子范围

`relu / add / bias_add`，用于验证：

- `dtype` 规则可复用
- `scope` 规则可复用
- `DDR / L2 / SMC` 搬运策略能否从 `matmul` 抽象出来

初始版本：输入输出全在 `L2` 走 `_p`；任一涉及 `DDR` 走 `_s`；`bias_add` 的 bias scope 也纳入版本选择；暂不做复杂 fusion。

### 5.6 matmul 已确认 / 待确认规则

已确认：

- 默认布局 `row_major`
- 已有手写高性能 matmul 样例
- `_s/_p` 路径含义已固化

待确认：

- 多核切分维度（M / N）
- 是否默认转置 B
- tile size 与 L2 占用之间的默认策略
- DMA chunk size 与性能模型

---

## 6. 变更摘要（按时间倒序）

### 第三轮：端到端最小闭环（A.2 接入 pipeline + A.5/A.6 起步）

> 详见 [learning.md §3 / §4](./learning.md)。

- 把 `C6678StoragePlan` 接入 [s_tir/pipeline.py](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py)：新增 `_c6678_specific_passes()`，在 `default_s_tir_pipeline` 的标准 pass 序列之前 `passes.extend(...)` 注入。c6678 之外的 target 不受影响（pass 内部按 `target.kind.name == "c6678"` 自我守卫）
- 新增 [s_tir/dlight/c6678/](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678) 包：
  - `__init__.py` 导出 `Matmul`
  - `base.py` 定义 `C6678ScheduleRule`（仅在 `target.kind.name == "c6678"` 生效）
  - `matmul.py` 提供极简 `[S,S,R]` matmul schedule：tile + reorder + parallel + auto unroll 注解
- 在 [s_tir/dlight/__init__.py](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/__init__.py) 注册新子包 `from . import c6678`
- 在 [Test4dsp/](file:///home/tangqingyun/tvm/Test4dsp) 下：
  - 新增 [tests/probe_c6678_build_baseline.py](file:///home/tangqingyun/tvm/Test4dsp/tests/probe_c6678_build_baseline.py)：探测 `tvm.tirx.build(target="c6678")` 是否端到端走通，并单独打印 A.2 attrs
  - 新增 [generate_c6678_matmul_via_build.py](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul_via_build.py)：fp32 128x128x128 matmul → `dlight.c6678.Matmul` schedule → `tvm.tirx.build` → 打印 scheduled TIR 与 C 源码，作为新路径的端到端 demo
- 修复 [tests/test_c6678_storage_plan.py](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_storage_plan.py) 中 `T.block` → `T.sblock` 的 TVMScript 方言差异

#### 设计要点

1. **TVMScript 方言**：本仓库 fork 把 `T.block(...)` 重命名为 `T.sblock(...)`（见 [tirx/script/builder/ir.py:371](file:///home/tangqingyun/tvm/python/tvm/tirx/script/builder/ir.py#L371)），写新测试时必须用 `T.sblock`
2. **`tvm.tirx.build` 已经能跑 c6678**：`is_host_func` 在 [tirx/build.py](file:///home/tangqingyun/tvm/python/tvm/tirx/build.py) 把 `c6678` 当作 host kind，通过 `target.build.c6678` 路由到 [codegen_c6678.cc](file:///home/tangqingyun/tvm/src/target/source/codegen_c6678.cc) `BuildC6678`
3. **A.2 在 pipeline 里只写 attrs，不影响 lower**：`c6678.storage_plan` 是 `func.attrs` 上的元数据，下游 lower / split_host_device_mods 不消费它，因此最终源码不会因为 A.2 的接入而改变；这是有意设计，让"识别"与"改写"分阶段交付
4. **`sch.parallel` 暂时是占位**：codegen 把所有循环输出为 serial，真正的多核派发由后续 `C6678MulticoreLower` 完成；annotate `pragma_auto_unroll_max_step` 同理，要等到 unroll pass 接入后才真正生效
5. **极简 schedule 优先打通端到端**：本轮 `Matmul` 只做 tile + reorder + parallel + unroll 注解，cache_read/DMA/tensorize 全部留给 A.3~A.5

### 第二轮：A-β 规则下沉（A.1 + A.2 雏形）

> 详见 [learning.md §3 路线图 §A.1/A.2](./learning.md)。

- 新增 [c6678_config.py](file:///home/tangqingyun/tvm/python/tvm/tirx/c6678_config.py)：`C6678Config + from_target`，c6678 硬件参数的"单一事实源"
- 新增 [c6678_storage_plan.py](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_storage_plan.py)：`C6678StoragePlan` pass，按 buffer scope 给 PrimFunc 写入 `c6678.storage_plan` 属性，**只识别不改写**
- [contrib/c6678.py](file:///home/tangqingyun/tvm/python/tvm/contrib/c6678.py) 中的硬件常量改为 re-export 自 `tirx.c6678_config`，避免规则两份
- 新增 [test_c6678_storage_plan.py](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_storage_plan.py) 作为最小冒烟脚本
- §8.5 规则归宿表中 per-core L2 / SMC / DDR / DMA scope / DMA 上限 5 行已升级，注明指向 `C6678Config` 与 `C6678StoragePlan`，最终落点改为 `multicore lowering` / `legality pass A.3`

#### 设计要点

1. **Python 侧 fallback 非常关键**：`Target("c6678").attrs` 不会自动带上 C++ 的 `DefaultValue`，所以 `C6678Config.from_target` 在 `_read_attr` 里强制走 `_C6678_DEFAULT_ATTRS` 兜底
2. **L2 是 per-core 资源**：A.2 plan 表里把 L2 buffer 的 `region_base/region_end` 暂时填 `-1`，等 multicore lowering 拿到 `core_id` 再填回；这意味着 `C6678MulticoreLower` 必须在 `C6678StoragePlan` 之后运行
3. **`global` 暂时按 DDR 处理**：默认 scope 在 TVM 中即"无标注"，对 6678 来说最安全的归宿就是 DDR
4. **不接 pipeline、不改 IR**：本轮 pass 只做 `func.with_attr(...)`，不会触发任何 buffer 重写、循环改写、DMA 注入

### 第一轮：MVP + Target Attrs 落地

- `c6678` target 硬件 attrs 第一批默认值已写入 [target_kind.cc](file:///home/tangqingyun/tvm/src/target/target_kind.cc)：覆盖 `core_num / core_freq_mhz / l1_size / l2_size / l2_base_core0 / l2_core_stride / smc_base / smc_size / ddr_base / ddr_size / dma_align_bytes / dma_burst_bytes / dma_max_transfer / vector_bytes`
- 新增 [contrib/c6678.py](file:///home/tangqingyun/tvm/python/tvm/contrib/c6678.py) 作为 6678 matmul MVP 的 Python 入口；关键 API：`build_matmul_module / build_matmul_source / get_l2_address_range / iter_cores_from_mask / validate_dma_path`
- [tirx/build.py](file:///home/tangqingyun/tvm/python/tvm/tirx/build.py) 已能产出 `c6678` 类型 `CSourceModule`，配套 [codegen_c6678.{h,cc}](file:///home/tangqingyun/tvm/src/target/source/codegen_c6678.cc)
- `Test4dsp/` 下新增/更新示例与轻量测试：`generate_c6678_matmul.py` / `tests/test_c6678_matmul_codegen.py`

---

## 7. 历史阶段计划（保留参考）

> 当前实际推进按 `learning.md` 中的 §3《调度优化框架路线图》执行；本节是早期文档中的阶段划分，保留以便理解早期决策。

| 阶段 | 目标 | 重点工作 |
|---|---|---|
| 1. 补齐 target 能力 | 让 `c6678` 具备硬件描述能力 | 硬件 attrs / DMA / 对齐 / vector width / per-core 地址查询 / `core_mask` 可读 |
| 2. 建立存储层级与 DMA 语义 | `DDR/L2/SMC` 下沉为编译器可见语义 | storage scope / DMA copy 表达 / legality 规则 / 多核同步语义 |
| 3. 构建 6678 专用 lowering pass | 从模板源码生成到正式 lowering | storage planning / DMA insertion / software pipeline / canonicalization / correctness / `core_mask` 解析 / 裸 C 输出 |
| 4. 建立首批可复用算子框架 | 不只做 matmul | 固化 matmul 的 tile/DMA/storage/多核 + elementwise 复用 |
| 5. 建立参数化模板库 | 单 matmul 扩展为专家模板 | 参数化 `tile/block/vector/dma chunk` + bias/activation/transpose + 单核/多核分离 |
| 6. 规则式模板匹配 | 根据特征自动选模板 | shape / scope / 硬件 / 算术强度 / 算子类别特征 |
| 7. 搜索空间与性能模型 | 升级为带优化能力的自动选择系统 | 搜索空间离散化 + legality 剪枝 + 计算/搬运/重叠性能模型 |
| 8. 接真实板卡 runner | 走向 auto-tuning | 编译/下发/执行/取回 + 经验数据库 |
| 9. 统一最终用户入口 | 标准 + 高层 DSL | 打通 `tvm.build(target="c6678")` 正式路径，统一 ABI |


### 4.2 ElementGreaterEqual 与 LSTM extern 的接入边界

- `ElementGreaterEqual` 已新增 `dlight.c6678.ElementGreaterEqual`，当前只覆盖 same-shape `float32 >= float32 -> bool`，生成 C 入口的 bool storage 是 `int8_t*`。这与 BSP 参考实现中的 `bool*` 需要后续统一 ABI；scalar/broadcast、输出 DMA store 和 BSP `get_l2_addr` 指针化还未做。
- `LSTM` 当前以 extern composite wrapper 形式接入：TVMScript 中显式 `call_extern("fp_lstm_s", ...)`，生成 bare-C wrapper。它不是自动 LSTM pattern matcher，不能宣称已完成 LSTM 自动调度。


### 4.3 ElementGreaterEqual 输入 DMA staging + 1D L2 compact

- 新增 `l2_dma_block_elems` helper，ElementGreaterEqual 根据 L2 容量计算 1D tile 大小。
- ElementGreaterEqual schedule 已插入 A/B 输入的 `cache_read("global.l2")` 和 `c6678.dma_load="dma_trans"`。
- `C6678DMALower` 的 1D `dma_trans` 已修正为使用 tile-dependent offset，并对 tail tile 使用 `min(extent, shape - offset)` 计算 `size_bytes`。
- `C6678DMALower` 内部已完成 1D L2 compact remap：DMA 目标使用 compact L2 offset，后续 L2 load/store 使用 tile-local index。
- 当前回归已验证源码包含 `dma_trans`、tile-dependent 输入地址、compact L2 staging storage、tile-local L2 index 和多核派发。
- 遗留：输出 DMA store、BSP `get_l2_addr(core_id)` 指针化、broadcast/scalar GreaterEqual 尚未完成。
