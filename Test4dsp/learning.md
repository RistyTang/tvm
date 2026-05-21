# TVM 当前项目架构解析与 6678DSP 开发手册

本文档用于把当前 `tvm/Test4dsp` 相关背景、设计目标、开发约束、已落地能力与标准开发流程整理成一份可直接交接给另一个 AI 模型或开发者的开发手册。

目标是让接手者在最短时间内理解：

1. 当前项目基于什么 TVM 结构开发
2. 6678DSP 支持现在做到什么程度
3. 后续开发必须遵循什么环境与命令规则
4. 标准开发流程应该怎么走

---

## 1. 项目背景

本项目是基于 Apache TVM 的深度定制版本。与上游 TVM 相比，传统 TIR 被拆分成了两个主要部分：

- `s_tir`：偏调度、MetaSchedule、后端代码降级
- `tirx`：偏 TIR 基础抽象、AST、内置算子、控制流与构建流程

当前目标不是做通用 DSP 编译器，而是基于 TVM 为 **TI C6678 DSP** 增加一条可持续演进的专用开发路径，优先从 `matmul` 单算子切入，再扩展到其余算子。

---

## 2. TVM 框架与仓库关键目录

### 2.1 TVM 主目录结构

```text
tvm/
├── 3rdparty/       # 第三方依赖
├── apps/           # 应用与部署示例
├── build/          # 编译产物
├── ci/             # CI 配置
├── cmake/          # CMake 配置
├── docs/           # 文档
├── include/        # 对外头文件
│   └── tvm/
│       ├── s_tir/
│       └── tirx/
├── python/         # Python 前端入口
│   └── tvm/
│       ├── s_tir/
│       └── tirx/
├── src/            # C++ 核心实现
│   ├── arith/
│   ├── ir/
│   ├── runtime/
│   ├── script/
│   ├── s_tir/
│   ├── tirx/
│   ├── support/
│   ├── target/
│   ├── te/
│   └── topi/
├── tests/
└── Test4dsp/       # 当前 6678DSP 相关开发与验证目录
```

### 2.2 与 6678DSP 支持最相关的目录

#### 1. `src/target/`

负责 target kind 注册、target 属性定义、代码生成器入口。

当前 6678 相关关键文件：

- `src/target/target_kind.cc`
- `src/target/source/codegen_c6678.cc`
- `src/target/source/codegen_c6678.h`

#### 2. `python/tvm/tirx/`

负责 Python 侧构建流程与 `tirx.build` 主路径。

当前关键文件：

- `python/tvm/tirx/build.py`

#### 3. `python/tvm/contrib/`

适合放置目标平台的 Python 辅助模块。

当前新增文件：

- `python/tvm/contrib/c6678.py`

这个模块是当前 `6678 matmul MVP` 的 Python 入口。

#### 4. `Test4dsp/`

当前开发者自己的 DSP 方向实验目录。

当前关键文件：

- `Test4dsp/learning.md`
- `Test4dsp/generate_c6678_matmul.py`
- `Test4dsp/tests/test_c6678_matmul_codegen.py`
- `Test4dsp/tests/test_my_ops.py`
- `Test4dsp/src/total_test.c`

---

## 3. 当前 6678DSP 支持现状

### 3.1 已完成的基础能力

当前仓库中，`c6678` 已经不是一个空名字，而是已经具备以下基础：

1. 已定义 `c6678 target`
2. 已接入 `target.build.c6678`
3. 已能通过 `tirx.build` 生成 `c6678` 类型的 C source module
4. 已在 Python 侧新增 `tvm.contrib.c6678` 作为 6678 matmul MVP 生成入口

### 3.2 当前新增的 6678 target 属性

`src/target/target_kind.cc` 中已经为 `c6678` 增加了第一批硬件属性入口：

- `core_num`
- `l1_size`
- `l2_size`
- `smc_size`
- `dma_align_bytes`
- `dma_burst_bytes`
- `dma_max_transfer`
- `vector_bytes`
- `workspace-byte-alignment`
- `constants-byte-alignment`

这些属性的作用：

- 为后续模板匹配提供硬件特征输入
- 为 DMA legality check 预留参数通道
- 为 block size 与内存层级规则选择提供依据

### 3.3 当前 MVP 生成能力

当前 `python/tvm/contrib/c6678.py` 提供了：

- `build_matmul_module(...)`
- `build_matmul_source(...)`

其定位不是完整通用 lowering，而是：

- 基于 Python 参数与 target attrs
- 直接生成 6678 风格的 C source module
- 优先解决“用户可生成可读、可集成、可继续演进的 6678 matmul 源码”

---

## 4. 开发环境与命令白名单

### 4.1 开发环境

#### 必须遵循的规则

**所有 Python 类命令必须运行在 `conda tvm_env` 环境中。**

也就是说，凡是涉及以下动作：

- `python xxx.py`
- `python - <<'PY' ... PY`
- `pytest`
- 任何依赖 `tvm / tvm_ffi / numpy` 的脚本

都必须先执行：

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh
conda activate tvm_env
```

否则很容易出现：

- `tvm_ffi` 导入异常
- `core` 循环导入异常
- 找不到 `tvm.contrib`
- Python 与本地编译产物不匹配

### 4.2 推荐的 PYTHONPATH

执行本仓库相关 Python 脚本时，推荐统一设置：

```bash
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python
```

完整推荐模板：

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python your_script.py
```

### 4.3 命令白名单

以下命令是当前项目中已验证、建议优先使用的“白名单命令模板”。

#### 1. 运行 Python 脚本

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python /home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul.py
```

#### 2. 运行内联 Python 验证

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

#### 3. 直接执行测试函数进行轻量验证

当 `pytest` 被 TVM 的 plugin 初始化拦住时，优先用这种方式：

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

#### 4. 读取生成源码

```bash
source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \
conda activate tvm_env && \
PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \
python - <<'PY'
from tvm.contrib import c6678
mod = c6678.build_matmul_module(
    M=64, N=64, K=64,
    dtype='float32',
    target='c6678',
    a_scope='ddr', b_scope='ddr', c_scope='ddr'
)
print(mod.inspect_source('c6678'))
PY
```

### 4.4 当前不推荐直接使用的方式

#### 1. 不要在未激活 `tvm_env` 的环境中运行 Python

错误示例：

```bash
python Test4dsp/tests/test_my_ops.py
```

#### 2. 不要默认依赖 `pytest` 直接跑通所有用例

原因：

- 当前环境里 TVM testing plugin 会在初始化时检查可用 target
- 由于本地 TVM 构建不一定启用了 `llvm/cuda/opencl/...`，可能在 plugin 阶段就报错

所以当前更稳妥的做法是：

- 优先直接执行测试函数
- 或者只运行不依赖 TVM testing plugin 的脚本

---

## 5. 项目硬件与运行时假设

当前对 C6678 的已知约束与假设如下：

1. 6678 有 8 个 CPU 核心
2. 每个核心有独立的 `L1` 和 `L2`
3. 存在核间共享空间 `SMC`
4. 读写速度关系大致为：`L2 > SMC > DDR`
5. 当前重点只考虑：`DDR / SMC / L2`
6. DMA 同步搬运函数默认是：

```c
void dma_trans(void *src, void *dst, int size)
```

7. 多核同步默认使用：

```c
void C6678E_SyncN(int core_num, int logic_core_id)
```

---

## 6. 总体设计目标

### 6.1 长期目标

设计一个“基于特征分类和专家模版”的 TVM 调度优化框架，用于为 6678DSP 自动生成经过优化的代码。

长期产出包括：

1. 一套面向 6678 的参数化调度模板库
2. 自动化模板匹配与搜索空间生成
3. 后续接入性能模型与实测调优

### 6.2 短期目标

短期先只聚焦 `matmul`。

目标是让用户可以写出接近 DSL 风格的 Python 配置，然后由系统自动生成 6678 风格的目标代码。

例如用户表达：

```python
input0 = ...   # 位于 ddr
input1 = ...   # 位于 ddr 或 l2
output = my_matmul(input0, input1, K, M, N)
```

系统应当根据：

- `dtype`
- `A/B/C` 的 scope
- 是否多核
- 是否需要 DMA 搬运

自动选择生成的算子版本。

---

## 7. 当前算子选择规则

这是当前已经实现、必须遵守的生成规则。

### 7.1 dtype 到前缀的映射

| dtype | 前缀 |
|---|---|
| `float32` | `fp` |
| `float64` | `dp` |
| `int8` | `i8` |
| `int16` | `i16` |
| `int32` | `i32` |

### 7.2 scope 到最终算子版本的映射

#### 规则 A：`A/B/C` 全在 `L2`

生成：

- `${prefix}_matmul_fusion_p`

含义：

- 认为数据已经位于片上
- 不需要通过 `_s` 路径进行 DDR -> L2 的 DMA 搬运
- 适合纯片上执行场景

#### 规则 B：其余情况，包括 `DDR`

生成：

- `${prefix}_matmul_fusion_s`

含义：

- 默认需要走带 DMA 搬运的多核 / 分块路径
- 因为 `_s` 版本里包含片外到片上的搬运逻辑，能更接近高性能实现

### 7.3 当前实现细节

当前源码生成后：

- **只公开一个最终函数名**
- 非目标版本不会作为公开最终算子出现
- 内部允许保留 `static *_impl` 实现函数

例如：

#### 情况 1：`dtype=float32` 且 `A/B/C` 在 DDR

公开最终算子：

```c
void fp_matmul_fusion_s(...)
```

#### 情况 2：`dtype=float32` 且 `A/B/C` 全在 L2

公开最终算子：

```c
void fp_matmul_fusion_p(...)
```

---

## 8. 当前已落地的 matmul MVP 设计

### 8.1 当前生成器的定位

`python/tvm/contrib/c6678.py` 不是通用 compiler pass，而是一个 **目标平台专用源码生成器**。

它完成的事情是：

1. 解析 Python 输入参数
2. 读取 `c6678 target attrs`
3. 根据 `dtype + scope` 决定目标最终算子版本
4. 生成一个 `c6678` 类型的 `CSourceModule`
5. 允许通过 `inspect_source('c6678')` 直接拿到源码文本

### 8.2 当前源码中包含的关键结构

当前生成的源码中，可能包含或使用以下结构：

- `C6678MatmulConfig`
- `dma_trans_2d`
- `copy_2d_by_scope`
- `load_row_major_tile`
- `load_transposed_tile`
- `store_row_major_tile`
- `*_matmul_block_kernel`
- `*_matmul_fusion_p_impl`
- `*_matmul_fusion_s_impl`
- 最终公开的 `*_matmul_fusion_p` 或 `*_matmul_fusion_s`

### 8.3 当前能力边界

当前版本还不是完整自动调度器，边界如下：

1. 还没有把通用 `tvm.build(target='c6678')` 自动识别 matmul 的全链路打通
2. 还没有把 `DDR/L2/SMC` 变成完整通用的 IR pass
3. 还没有构建真正的性能模型
4. 还没有接真实板卡 runner
5. 还没有做自动搜索空间生成

但它已经提供了一个稳定起点：

- 可以根据规则生成正确命名的最终目标算子
- 可以沉淀对其他算子的模板式开发方法

---

## 9. 预期开发步骤

后续要从 `matmul MVP` 演进到“面向 6678 的 TVM 调度优化框架”，建议按以下阶段推进。

### 阶段 1：补齐 target 能力

目标：让 `c6678` 不只是 target 名字，而是具备足够的硬件描述能力。

重点工作：

1. 继续补充硬件 attrs
2. 明确 DMA、对齐、bank、vector width 等约束
3. 整理 target attrs 的默认值与覆盖方式

### 阶段 2：建立存储层级与 DMA 语义

目标：把 `DDR/L2/SMC` 从“Python 参数”逐步下沉为编译器可见语义。

重点工作：

1. 定义 storage scope
2. 定义 DMA copy 表达方式
3. 加入 legality 规则
4. 统一多核同步语义

### 阶段 3：构建 6678 专用 lowering passes

目标：从“模板式源码生成”演进到“编译链内的正式 lowering”。

重点工作：

1. storage planning
2. DMA insertion
3. software pipeline / double buffer
4. target-specific canonicalization
5. correctness pass

### 阶段 4：建立参数化模板库

目标：把单一 `matmul` 模板扩展成专家模板体系。

重点工作：

1. 参数化 `tile_m / tile_n / tile_k`
2. 参数化 `block_size / vector_width / dma chunk`
3. 支持 bias / activation / transpose / storage hint
4. 分离单核与多核模板

### 阶段 5：做规则式模板匹配

目标：根据问题特征自动选择模板。

重点工作：

1. shape 特征
2. scope 特征
3. 硬件特征
4. 算术强度与访存模式特征

### 阶段 6：搜索空间与性能模型

目标：从规则系统升级为带优化能力的自动选择系统。

重点工作：

1. 搜索空间离散化
2. legality 剪枝
3. 计算 / 搬运 / 重叠 三部分性能模型
4. bandwidth-bound / compute-bound 分类

### 阶段 7：接真实板卡 runner

目标：让模板与模型进入“可测量、可迭代”的状态。

重点工作：

1. 编译
2. 下发
3. 执行
4. 取回时间
5. 建立经验数据库

---

## 10. 标准开发流程 SOP

下面是当前推荐给 AI 模型或开发者使用的标准开发流程。

### SOP-0：先确认目标类型

开发前先判断本次任务属于哪类：

1. `target 能力补充`
2. `源码生成器增强`
3. `新算子模板接入`
4. `测试与验证`
5. `编译链正式下沉`

### SOP-1：先看现状，再定修改层级

如果需求只是快速验证某个算子的 6678 代码结构：

- 优先改 `python/tvm/contrib/c6678.py`
- 不要一上来就改 `tirx / s_tir / codegen_c6678` 全链路

如果需求已经明确要接入正式 lowering：

- 再考虑往 `tirx / s_tir / src/target` 下沉

### SOP-2：优先做单算子 MVP

新增算子时，优先按以下顺序推进：

1. 增加 Python 入口
2. 增加 config 结构体
3. 增加模板实现
4. 增加示例脚本
5. 增加轻量测试
6. 最后再考虑编译链融合

### SOP-3：新增算子的推荐文件组织

如果要扩展新算子，例如 `conv`：

1. 在 `python/tvm/contrib/c6678.py` 中新增入口
   - 例如 `build_conv_module(...)`
2. 在 `Test4dsp/` 下新增生成示例
   - 例如 `generate_c6678_conv.py`
3. 在 `Test4dsp/tests/` 下新增轻量测试
   - 例如 `test_c6678_conv_codegen.py`
4. 在 `learning.md` 中追加该算子的规则说明与版本选择策略

### SOP-4：每次开发都要同步记录以下内容

每次修改后，至少补充以下信息到文档或变更说明：

1. 涉及文件
2. 每个文件的核心修改
3. 修改目的
4. 影响范围
5. 新增规则
6. 后续扩展建议

### SOP-5：优先使用轻量验证，而不是默认跑全量 pytest

当前最稳的验证策略：

1. 先运行内联 Python 检查是否成功生成 `CSourceModule`
2. 再读取 `inspect_source('c6678')` 检查源码结构
3. 如果有测试函数，直接调用测试函数
4. 等环境更稳定后再接入标准 pytest 流程

### SOP-6：关于最终导出函数命名的规则

新增算子时必须尽量遵循：

- 根据 `dtype` 决定前缀
- 根据 `scope` / 执行路径决定最终版本后缀
- 只公开一个最终目标算子名
- 其余辅助函数应当是内部 helper 或 `static impl`

### SOP-7：关于 ABI 的长期要求

项目有一个非常重要的长期约束：

生成的代码最终应当符合如下风格：

```c
void funcname(void* self_handle, void* args, int32_t num_args, void* result)
```

也就是说：

- 最终要与当前更业务化的模板生成方式逐步收敛
- 不应长期停留在纯实验接口
- 当前 MVP 可先保留简化接口，但后续正式接入编译链时必须考虑 ABI 收敛

---

## 11. 当前已验证的调用示例

### 11.1 生成 DDR 路径的 `float32` matmul

```python
from tvm.contrib import c6678

mod = c6678.build_matmul_module(
    M=256,
    N=256,
    K=256,
    dtype='float32',
    target='c6678',
    a_scope='ddr',
    b_scope='ddr',
    c_scope='ddr',
)

source = mod.inspect_source('c6678')
```

预期：

- 公开最终函数为 `fp_matmul_fusion_s`

### 11.2 生成全 L2 路径的 `float32` matmul

```python
from tvm.contrib import c6678

mod = c6678.build_matmul_module(
    M=256,
    N=256,
    K=256,
    dtype='float32',
    target='c6678',
    a_scope='l2',
    b_scope='l2',
    c_scope='l2',
    use_multicore=False,
)

source = mod.inspect_source('c6678')
```

预期：

- 公开最终函数为 `fp_matmul_fusion_p`

---

## 12. 当前已知问题与注意事项

### 12.1 `Test4dsp/tests/test_my_ops.py` 当前不是本轮主链路

当前这个文件的报错表明：

- `load_lib_ctypes(...)` 的签名已经变化
- 不能再按旧方式只传一个 `.so` 名称

如果后续要修这个测试，需要重新看：

- `tvm_ffi.libinfo.load_lib_ctypes(...)`
- 它当前要求的 `package / target_name / mode` 参数

### 12.2 `pytest` 可能因为 TVM plugin 初始化失败

如果看到类似：

- `None of the following targets are supported by this build of TVM`

说明不是本次业务逻辑必然有错，而是测试环境初始化时缺少某些 target 支持。

优先使用轻量验证方案。

---

## 13. 交接给另一个 AI 时应优先阅读的文件

建议接手者按以下顺序阅读：

1. `tvm/Test4dsp/learning.md`
2. `tvm/python/tvm/contrib/c6678.py`
3. `tvm/Test4dsp/generate_c6678_matmul.py`
4. `tvm/Test4dsp/tests/test_c6678_matmul_codegen.py`
5. `tvm/src/target/target_kind.cc`
6. `tvm/src/target/source/codegen_c6678.cc`
7. `tvm/python/tvm/tirx/build.py`

---

## 14. 一句话总结

当前项目已经完成了 **6678DSP 的 matmul MVP 源码生成路径**：

- `dtype` 决定前缀
- `scope` 决定最终版本
- `DDR -> _s`
- `全 L2 -> _p`
- 最终只公开一个目标算子

后续开发应当先沿着这个模板式路径扩算子，再逐步下沉到正式的 TVM 编译链与自动调优框架中。
