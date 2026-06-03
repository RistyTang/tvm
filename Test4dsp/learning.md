# TVM 6678DSP 开发主文档（精简版）

> 本仓库为 Apache TVM 的深度定制版本，目标是为 **TI C6678 DSP** 增加一条可持续演进的专用开发路径，先从 `matmul` 切入，再扩展到其余算子。
>
> 本文只承载**架构、硬件假设、当前状态、调度优化路线图、接手索引**；其余可移动的内容已拆到附录：
> - **流程 / 历史 / 命令白名单 / 示例** → 见 [`dev_history.md`](./dev_history.md)
> - **API 参考（target attrs、`tirx.c6678_config`、`C6678StoragePlan`、`contrib.c6678`）** → 见 [`api_reference.md`](./api_reference.md)
>
> 接手 agent 推荐阅读顺序：本文 §1~§3 → §4 路线图 → `api_reference.md` → 需要写代码时再翻 `dev_history.md` SOP。

---

## 1. 项目背景与代码地图

本仓库与上游 TVM 的核心区别：传统 TIR 被拆分成两半。

| 模块 | 定位 |
|---|---|
| `s_tir` | 偏调度、MetaSchedule、后端代码降级（pipeline、dlight 模板） |
| `tirx` | 偏 TIR 基础抽象、AST、内置算子、构建流程（`tirx.build`、analysis、transform） |

与 6678 支持最相关的目录：

| 路径 | 角色 |
|---|---|
| `src/target/target_kind.cc` | c6678 TargetKind + 14 项硬件 attrs DefaultValue |
| `src/target/source/codegen_c6678.{cc,h}` | `target.build.c6678` 注册（`BuildC6678`） |
| `python/tvm/tirx/build.py` | c6678 当前作为 host kind 路由 |
| `python/tvm/tirx/c6678_config.py` | **硬件常量唯一事实源**（A.1） |
| `python/tvm/tirx/transform/c6678_storage_plan.py` | C6678 storage planning pass（A.2） |
| `python/tvm/contrib/c6678.py` | matmul MVP 入口（薄壳，仅 re-export 常量） |
| `Test4dsp/` | 当前 DSP 方向实验目录（本文所在地） |

---

## 2. 硬件与运行时假设（精要）

完整地址表与 DMA 规则的历史推导见 `dev_history.md` §5；这里只列必须记住的硬要：

- **核心**：8 核，独立 L2，频率 1.25 GHz。
- **L2**：每核独享 1 MB，`core i` 落在 `0x10800000 + i * 0x01000000`。
- **SMC**：全核共享 8 MB，`0x0C000000 ~ 0x0C7FFFFF`。
- **DDR**：`0x80000000 ~ 0xFFFFFFFF`（开发假设上界）。
- **DMA**：仅在 `L2 / SMC / DDR` 间建模；同步、最大单次 `INT_MAX`、`dma_align_bytes = 64`。
  当前阶段**不考虑 L1D / L1P、异步、burst overlap**；已支持 `load_row_major_tile` 与 `dma_trans` 两类同步 DMA wrapper 的静态合法性校验，但 `dma_trans` 尚未被 softmax phase A 自动调度使用。
- **多核**：`core_mask` 8-bit 位图，bit i 表示 core i；`use_multicore=False` 强制 `0x01`。
- **算子命名**：`${prefix}_matmul_fusion_${variant}`，`prefix ∈ {fp,dp,i8,i16,i32}`，
  `variant = p`（A/B/C 全 L2）或 `s`（含 DMA）。

完整 target attrs 表见 [`api_reference.md` §1](./api_reference.md#1-c6678-target-属性表)。

---

## 3. 当前能力快照

### 3.1 已落地

| 能力 | 入口 |
|---|---|
| c6678 TargetKind + 14 项 attrs DefaultValue | `src/target/target_kind.cc` |
| `target.build.c6678` 注册 | `src/target/source/codegen_c6678.cc` |
| matmul MVP 字符串模板生成 | `tvm.contrib.c6678.build_matmul_module` |
| 硬件常量统一入口（A.1） | `tvm.tirx.c6678_config.from_target` |
| Storage planning pass 雏形（A.2） | `tvm.tirx.transform.C6678StoragePlan` |
| A.2 已挂入 `default_s_tir_pipeline` | `python/tvm/s_tir/pipeline.py::_c6678_specific_passes` |
| 极简 matmul schedule rule（A.5/A.6 起点） | `tvm.s_tir.dlight.c6678.Matmul` |
| **极简 softmax schedule rule（PR-S2 phase A）** | [`tvm.s_tir.dlight.c6678.Softmax`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py)（4 块：max → exp → sum → div，`parallel(outer)` + 逆序 `compute_at(outer, preserve_unit_loops=True)`） |
| **ElementGreaterEqual schedule rule（EGE-S1）** | [`tvm.s_tir.dlight.c6678.ElementGreaterEqual`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/elementwise.py)（单 `S/SS/...` injective block，store value 为 `tirx.GE`，按 L2 capacity split 外层，两个输入 `cache_read("global.l2")` 并标注 `dma_trans`，输出 bool 目前按 TVM C codegen 落为 `int8_t*`） |
| Bare-C entry signature（A.7） | `tvm.tirx.transform.C6678LowerEntry` + `python/tvm/s_tir/pipeline.py::_c6678_pre_packed_api_passes` |
| 多核派发 + SyncN（A.8） | `tvm.tirx.transform.C6678MulticoreLower`（与 A.7 同链路于 `_c6678_pre_packed_api_passes`） |
| **`scope='global'` AllocBuffer 自动加 `disable_lower_builtin` 注解（PR-S2 phase A）** | [`tvm.tirx.transform.C6678AnnotateGlobalAlloc`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) —— 紧跟 `C6678AnnotateL2Alloc` 之后，避开 c6678 bare-metal 没有 `device_id` AttrStmt 时 `LowerTVMBuiltin` 走 `TVMBackendAllocWorkspace` 的崩溃路径 |
| 端到端最小闭环 demo（matmul → schedule → tvm.tirx.build → C 源码） | `Test4dsp/generate_c6678_matmul_via_build.py` |
| **端到端 softmax demo（PR-S2 phase A）** | [`Test4dsp/generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) → 落盘 [`tests/generated_c6678_softmax_via_build.c`](file:///home/tangqingyun/tvm/Test4dsp/tests/generated_c6678_softmax_via_build.c)（**1925 chars 快照**） |
| 冒烟测试 | `Test4dsp/tests/test_c6678_storage_plan.py`、`test_c6678_matmul_codegen.py`、`test_c6678_end_to_end.py`（matmul 7/7）、`probe_c6678_build_baseline.py` |
| **PR-S2 phase A 形式化回归** | [`Test4dsp/tests/test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py)（softmax 7/7：bare-C 入口 / 多核派发+SyncN / `T_max[1]/T_exp[1024]/T_sum[1]` 栈数组（无 `TVMBackendAllocWorkspace`）/ 4 串行内层（ax1, ax1_1, ax1_2, ax1_3）/ `expf(` 直降 / 1925 chars 零差分快照） |
| **ElementGreaterEqual 形式化回归** | [`Test4dsp/tests/test_c6678_greater_equal_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_greater_equal_codegen.py)（same-shape `A >= B`，bare-C 入口 / `int8_t*` bool storage / 输入 `dma_trans` staging / 多核派发 + `C6678E_SyncN`） |
| **LSTM extern wrapper 回归** | [`Test4dsp/tests/test_c6678_lstm_extern_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_lstm_extern_codegen.py)（`lstm_fp32_extern(...)` 直接调用 BSP 复合核 `fp_lstm_s(Output, Input, Params, core_mask)`，不是完整 LSTM pattern matcher） |

> A.1/A.2 已通过冒烟测试；端到端 `tvm.tirx.build(target="c6678")` 已可在 fp32 128x128x128 matmul 上跑通并产出 C 源码。
> 验证脚本与命令模板见 `dev_history.md` §1~§3。

### 3.2 边界与已知未做

- `C6678StoragePlan` 现阶段仍只 `func.with_attr("c6678.storage_plan", ...)` —— 已挂入 pipeline，但下游 lower/host-device split 不消费该 attr，最终源码不感知；L2 buffer 的 `region_base/end` 占位 `-1`，等 `C6678MulticoreLower` 拿到 `core_id` 再回填。
- `dlight.c6678.Matmul` 仅做 tile + reorder + parallel 注解，没有 cache_read / DMA / tensorize；当前 codegen 把 `parallel` 输出为 serial loop，是占位，给 A.5 多核 lower 留接口。
- **A.6 进度（PR-S2 phase A 已落地，2026-05-25）**：除 `Matmul` 外，新增 [`dlight.c6678.Softmax`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py)（4 块 PrimFunc 形态：max → exp → sum → div；schedule 顺序与 `dlight/cpu/reduction.py` 对齐：`sch.parallel(outer)` 在前、`for bi in reversed(block_infos[:-1]): sch.compute_at(bi.block_rv, outer, preserve_unit_loops=True)` 在后）。当前阶段不引入 staging / DMA（phase B 再做），出码体现为 4 个串行内层 `for (ax1/ax1_1/ax1_2/ax1_3 in [0, INNER))` + 多核外层 + 末尾 `C6678E_SyncN`；BSP ABI 唯一新增依赖是 `extern float expf(float)`（C99 标准库符号，由 `tirx.exp` 默认 intrin lowering 直接降为 `expf(...)`，无需 c6678 专属 intrin）。Conv / reduce 仍走默认 codegen；`ElementGreaterEqual` 已有 dlight rule，可完成 same-shape `float32 >= float32 -> bool` 出码；输入侧 `cache_read("global.l2") + dma_trans` 与 1D L2 compact 已落地，broadcast、输出 DMA store、BSP `get_l2_addr` 指针化仍待补。
- `contrib/c6678.py` 仍是字符串模板，作为旧路径并存；新链路通过 `tvm.tirx.build` 直接出码。
- DMA legality（A.3）✅ 已落地（2026-05-24）：`python/tvm/tirx/transform/c6678_dma_legalize.py` 紧贴 `C6678DMALower` 之后挂入 `_c6678_post_schedule_passes`，对 `call_extern("load_row_major_tile", ...)` 与 `call_extern("dma_trans", ...)` 做纯静态校验（scope 白名单 / 正数性 / `rows*cols*elem_size ≤ dma_max_transfer` 或 `size_bytes ≤ dma_max_transfer` / 行对齐告警）；只读不改 IR，`via_build` 出码仍为 2345 chars。A.4 特征抽取已落地（`python/tvm/tirx/analysis/c6678_features.py`，未挂入 pipeline，留给 A.5 proper dispatcher 消费）。
- **当前 `tvm.tirx.build` 输出已脱去 TVM FFI wrapper 且接通多核派发（A.7 + A.8 已落地）**：c6678 host PrimFunc 现在以 `void matmul_fp32(float* A, float* B, float* C, int32_t core_mask)` 形态出码，`ForKind.PARALLEL` 已在 IR 层降为 `if (GetCoreNum > 0 && c6678_get_core_id >= 0) { ... }` 多核分块循环 + 末尾 `C6678E_SyncN(...)`，BSP 端可以 `extern void matmul_fp32(...)` 直接链接。
- **A.9（C6678DMALower）+ A.10 step1（dlight schedule 端 cache_read/annotate）联合 PR 已落地（✅，2026-05-24）**：
  - `dlight.c6678.Matmul` schedule 端已注入 `cache_read("global.l2") + compute_at(k_outer, preserve_unit_loops=True)` + `sch.annotate("c6678.dma_load"="load_row_major_tile", "c6678.src_buffer", "c6678.src_scope")`。
  - A.9 已拆为两段 pass（见 §4.8.3 第 3 条）：① `C6678DMALower` 在 `ConvertBlocksToOpaque` **之前**（`PlanAndUpdateBufferAllocationLocation` 之后）执行，把带 `c6678.dma_load` 注解的 staging block 替换为 `Evaluate(call_extern("load_row_major_tile", src.data, dst.data, row0, col0, rows, cols, src_ld, sizeof_elem, src_scope))`；② `C6678AnnotateL2Alloc` 在 `StorageRewrite` **之后**、`LowerTVMBuiltin` **之前**执行，给 `scope=="global.l2"` 的 `AllocBuffer` 通过 `tirx.decl_buffer(scope="global", ...)` 重建 `Buffer.data` 的 `PointerType.storage_scope = "global"`，然后第二轮 `ir_transform` 把 body 中所有 `BufferLoad/BufferStore` 的 `buffer` 字段一并替换为新 Buffer，再用 `stmt_functor.substitute` 把 PrimExpr 里的 data Var 引用替换；同时给 `AllocBuffer` 留下 `disable_lower_builtin = True` 注解作为 `LowerTVMBuiltin` 端的兜底。
  - 实测：`generated_c6678_matmul_via_build.c` 出码 2345 chars（基线 1110，A.8 落地后 2075，A.9+A.10 step1 后 2309，A.9 follow-up 修复后 2345）；A 段 DMA 出 `load_row_major_tile((&(A[0])), (&(A_global_l2[0])), ...)`、B 段 DMA 出 `load_row_major_tile((&(B[0])), (&(A_global_l2[16384])), ...)`，B 的目标地址正确指向合并后 alloc 的 B 段；3 个回归测试 `test_c6678_storage_plan / probe_c6678_build_baseline / test_c6678_matmul_codegen` 全部 PASS（baseline 1110 chars 不变）。
  - **Follow-up 已闭环**：`StorageRewrite` 把 `A_global.l2` 与 `B_global.l2` 合并到同一块 alloc + 用 elem_offset=16384 区分，由 `C6678DMALower::_build_dma_call` 用 `src_buf.access_ptr("r") / dst_buf.access_ptr("w")` 替代裸 `.data`，让 `StorageRewrite` 看见数据流自动合成偏移；后续 `LowerIntrin` 把 `tvm_access_ptr` 降级为 `address_of(BufferLoad)`，C codegen 出 `&A_global_l2[16384]` 形态。

---

## 4. 调度优化框架路线图（特征分类 + 专家模板）

> 仅靠 `contrib/c6678.py` 的字符串模板无法承载"算子调优"。要实现"特征分类 + 专家模板派发 + schedule 落地"，必须把规则迁移到 `tirx`/`s_tir` 真正的 IR pass。本节是这一迁移的总路线，作为后续各 PR 的母提案。

### 4.1 终极形态

> 本节给两张图：**图 A** 是路线图终态（含尚未落地的 A.10 step2，方便接手者一眼看到目标），**图 B** 是 2026-05-24 实测的当前真实管线（与 [`pipeline.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py) 一一对应）。两图的差距即 §4.2 表里仍标 ⏳/🟡 的格子，§4.1.4 给出对照。A.4 `C6678Features` 模块已落地（[`c6678_features.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/analysis/c6678_features.py)），但作为只读分析 pass **暂不挂入** `default_s_tir_pipeline`，由 A.5 dispatcher（[`dispatcher.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/dispatcher.py)）在 schedule 阶段直接调用。A.3 [`C6678DMALegalize`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_legalize.py) 已落地并挂入 `_c6678_post_schedule_passes`（紧跟 A.9 之后）。

#### 4.1.1 图 A：终态愿景

```
@T.prim_func + Target("c6678")          ← 用户唯一入口：tvm.build(mod, target="c6678")
        │
        ▼
[A.4 C6678Features]   ──> features(shape/dtype/scope/loops/reduce_axis/...)
        │
        ▼
[A.5 dlight.c6678 分发器] ──> 选 schedule 模板（含 tile / cache_read / dma_copy / parallel）
        │
        ▼
[schedule 应用]   ──> 改写后的 PrimFunc + staging block annotations
        │
        ▼
┌────────── c6678 专属 IR pass 队列（按主流水线 4 个挂载点分布） ──────────┐
│  ① schedule 起点（pipeline 起始）                                      │
│     • A.2  C6678StoragePlan       —— 写 c6678.storage_plan attrs      │
│  ② 在 PlanAndUpdateBufferAllocLoc 之后、ConvertBlocksToOpaque 之前      │
│     • A.9  C6678DMALower          —— 三层 For/SBlock → load_row_major │
│     • A.3  C6678DMALegalize       —— scope/对齐/越界校验（只读，✅）   │
│  ③ 在 StorageRewrite 之后、LowerTVMBuiltin 之前                       │
│     • A.9  C6678AnnotateL2Alloc   —— PointerType.scope→"global"        │
│  ④ 在 SplitHostDevice/MergeSharedMem 之后、MakePackedAPI 之前          │
│     • A.7  C6678LowerEntry        —— bare-C entry + bypass packed API │
│     • A.8  C6678MulticoreLower    —— PARALLEL → CoreId/SyncN          │
└────────────────────────────────────────────────────────────────────────┘
        │
        ▼
codegen_c6678.cc → CSourceModule (.c 源码 + BSP extern 调用)
        │
        ▼
BSP 链接：extern void matmul_fp32(float*, float*, float*, int32_t core_mask)
```

#### 4.1.2 图 B：当前管线快照（2026-05-24 实测）

> 与 [`pipeline.py::default_s_tir_pipeline`](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py#L85-L194) 一一对应；c6678 专属 pass 用 `★` 标出，其它行是主流水线既有 pass。

```
[c6678 schedule 阶段]
  • dlight.c6678.Matmul.apply   ★ tile + reorder + cache_read("global.l2")
                                  + compute_at(k_outer) + sch.annotate(...)
                                  + parallel(i_outer)
        │  PrimFunc 已带 staging block annotation
        ▼
[default_s_tir_pipeline] —— 挂载点 ① schedule 起点
  • C6678StoragePlan            ★ A.2 ✅ 仅写 attrs，下游未消费
  • CanonicalizeLoop / LowerCrossThreadReduction / LowerInitBlock
  • PlanAndUpdateBufferAllocationLocation
        │
        ▼  —— 挂载点 ② post-schedule
  • C6678DMALower               ★ A.9 ✅ 三层 For/SBlockRealize → Evaluate(call_extern("load_row_major_tile",...))
        │
        ▼
  • ConvertBlocksToOpaque / LiftThreadBinding / ManifestSharedMemoryLocalStage
  • CompactBufferAllocation / LowerAutoCopy / UnifyThreadBinding
  • LowerMatchBuffer / Simplify / InjectPermutedLayout / AnnotateIrregularLoop
  • InjectSoftwarePipeline / TransformMmaBufferLayout
  • LowerOpaqueBlock / FlattenBuffer
  • BF16ComputeLegalize / NarrowDataType(32) / LoopPartition
  • VectorizeLoop / InjectVirtualThread / InjectDoubleBuffer
  • StorageRewrite           （注：会丢弃 AllocBuffer 上除 kVolatile 外的注解）
        │
        ▼  —— 挂载点 ③ post-StorageRewrite
  • C6678AnnotateL2Alloc        ★ A.9 ✅ 三步重写 buffer.scope="global.l2"→"global"
                                  + disable_lower_builtin=True
        │
        ▼
  • HoistIfThenElse / UnrollLoop / RenormalizeSplitPattern / Simplify / RemoveNoOp
  • RewriteUnsafeSelect / FP8ComputeLegalize / VerifyVTCMLimit / LowerVtcmAlloc
  • VerifyMemory / AnnotateEntryFunc
  • ThreadSync*3 / InferFragment / LowerThreadAllreduce
  • AnnotateDeviceRegions / SplitHostDevice / MergeSharedMemoryAllocations
        │
        ▼  —— 挂载点 ④ pre-MakePackedAPI
  • C6678LowerEntry             ★ A.7 ✅ 改 params/buffer_map/calling_conv=kCPackedFunc
  • C6678MulticoreLower         ★ A.8 ✅ PARALLEL→GetCoreNum/c6678_get_core_id/C6678E_SyncN
                                            + 入口追加 int32_t core_mask
        │
        ▼
  • MakePackedAPI               （c6678 host func 因 calling_conv != kDefault 直接早返回）
  • FP8StorageLegalize / BF16StorageLegalize / LowerDeviceKernelLaunch
        │
        ▼
codegen_c6678.cc → CSourceModule
        │
        ▼
generated_c6678_matmul_via_build.c (实测 2345 chars)
```

#### 4.1.3 BSP ABI 契约（已固化，BSP 端必须实现）

c6678 host PrimFunc 出码后对外契约固定为以下形态。BSP 工程链接此 `.c` 时必须提供下列 `extern` 符号（命名与 `generated_c6678_matmul.c` 旧字符串模板对齐，便于复用同一份 BSP 实现）：

**入口签名**：

```c
void matmul_fp32(float* A, float* B, float* C, int32_t core_mask);
```

- `A/B/C`：DDR 全局指针（出码顶部对应 `float*` 形参，由 [`C6678LowerEntry`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_lower_entry.py) 把 PrimFunc.params 改写为 `buffer.data` 直接产生）。
- `core_mask`：BSP 多核掩码（由 [`C6678MulticoreLower`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_multicore_lower.py) 末尾追加）。

**必须由 BSP 端提供的 `extern` 符号**：

| 符号 | 出处 pass | 语义（BSP 端实现责任） |
|---|---|---|
| `int32_t GetCoreNum(int32_t core_mask)` | A.8 `C6678MulticoreLower` | 由 `core_mask` 解出本组激活核数 |
| `int32_t c6678_get_core_id(int32_t core_mask)` | A.8 `C6678MulticoreLower` | 封装 BSP 端 `GetLogicCoreId(core_mask, DNUM)`，IR 侧不感知 `DNUM` 宏 |
| `void C6678E_SyncN(int32_t core_num, int32_t core_id)` | A.8 `C6678MulticoreLower` | 同步 N 核（替代 IR 侧不可用的 barrier） |
| `void load_row_major_tile(const void* src, void* dst, int32_t row0, int32_t col0, int32_t rows, int32_t cols, int32_t src_ld, int32_t elem_bytes, const char* src_scope)` | A.9 `C6678DMALower` | DMA 搬运一块 tile 到 L2 staging |
| `void dma_trans(void* src, void* dst, int32_t size_bytes)` | A.3/A.9 已支持合法性校验与 1D annotation lowering；softmax phase A 暂未自动使用 | 连续 1D 同步 DMA 搬运 wrapper |
| `void fp_matmul_fusion_p_(...)` | A.10 step2（待落地） | `_p_` 内层退化开关命中时调用，便于复用旧字符串模板的 SIMD/intrinsic 实现 |

> 关键设计决策：**IR 侧不出现任何 BSP 头宏**（`DNUM`、`LL2_*_BASE`、`#include <78NE/initial.h>` 等），所有平台细节封在 BSP 端 `extern` 函数后面；codegen 出来的 `.c` 是纯算术 + `extern` 调用，可被任何 BSP 工程直接 `extern` 链接。

#### 4.1.4 终态差距（图 A 比图 B 多了什么）

下表把图 A 中尚未在图 B 出现的格子，对应回 §4.2 的子任务编号，便于按行交付：

| 图 A 节点 | 图 B 现状 | §4.2 行 | 差距 |
|---|---|---|---|
| A.4 C6678Features | ✅ 已落地（2026-05-24） | A.4 ✅ | `python/tvm/tirx/analysis/c6678_features.py` 提供 `extract_features(func, target)`，输出 `C6678PrimFuncFeatures(blocks=tuple[C6678OpFeatures], config=C6678Config)`：每 block 抽 op_kind / dom_kind / dom_extents / dtype / read_bufs / write_bufs / flop_count_static / tile_hint / static_alloc_l2_bytes；冒烟 3 用例（matmul→gemm dom=(128,128,128) tile=(32,32,32) alloc=131072B flops=4194304；非 c6678 target→None；scale→elementwise dom=(128,128)）全过 |
| A.5 dlight.c6678 分发器（proper） | ✅ 已落地（2026-05-24） | A.5 ✅ | `python/tvm/s_tir/dlight/c6678/dispatcher.py` 提供 `ScheduleTemplate` 抽象 + `MatmulGemmTemplate` + `select_template(features) -> ScheduleTemplate \| None`；`Matmul.apply` 退化为薄壳（抽特征 → 选模板 → 模板就地改写 sch）；派发器自带 L2 容量门禁（按 `feats.config.l2_size` 卡更保守的读 buffer staging 全量 `static_alloc_l2_bytes`；128×128 matmul 为 131072B，与实际 `float A_global_l2[32768]` 对齐）；零差分验证：`via_build` 出码仍为 2345 chars |
| A.3 C6678DMALegalize | ✅ 已落地（2026-05-24） | A.3 ✅ | `python/tvm/tirx/transform/c6678_dma_legalize.py` 紧跟 A.9 `C6678DMALower` 之后挂入 `_c6678_post_schedule_passes`，对 `call_extern("load_row_major_tile", ...)` 与 `call_extern("dma_trans", ...)` 做纯静态校验（scope ∈ {global, global.l2, global.smc} / 正数性 / `rows*cols*elem_size` 或 `size_bytes` 不超过 `dma_max_transfer` / `cols*elem_size` 非 `dma_align_bytes` 整数倍仅 `C6678DMAAlignmentWarning`）；只读不改 IR，幂等（`c6678.dma_legalized` flag），非 c6678 target 直接跳过；冒烟 9 用例（合法 / 非法 scope / 超容量 / 行对齐告警 / 幂等 / 非 c6678 跳过）全过；端到端 `via_build` 出码仍为 2345 chars |
| A.10 step2 内层 `_p_` 退化开关 | 不存在 | A.10 🟡 | `dlight.c6678.Matmul` 当 tile 命中 `_p_` 形态时把内层换成 `call_extern("fp_matmul_fusion_p_",...)` |
| A.9 follow-up：B 的 DMA 第二参数 | ✅ 已修（2026-05-24） | §4.8.3 第 3 条 | 已把 `dst.data` 替换为 `dst.access_ptr("w")`，`StorageRewrite` 自动合成 `&A_global_l2[16384]`；实测 `via_build` 出码 2345 chars（原 2309，+36），3 回归测试通过 |
| **PR-S2 phase A：dlight.c6678.Softmax + C6678AnnotateGlobalAlloc** | ✅ 已落地（2026-05-25） | §4.9（新增）| dlight 端新增 [`Softmax`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py)（4 块 PrimFunc：max → exp → sum → div，`parallel(outer)` + 逆序 `compute_at`）；pipeline 端新增 [`C6678AnnotateGlobalAlloc`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) 紧跟 `C6678AnnotateL2Alloc` 之后；端到端 demo [`generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) 实测 1925 chars 出码（含 `expf(...)`、`T_max[1]/T_exp[1024]/T_sum[1]` 栈数组、4 串行内层、多核派发 + `C6678E_SyncN`）；matmul 端零差分（仍 2345 chars） |

> 最终用户仍只写 `tvm.build(mod, target="c6678")` 一行；所有上述差距对最终用户透明。

### 4.2 子任务拆解

| 编号 | 名称 | 主要落点 | 状态 |
|---|---|---|---|
| A.1 | C6678 attrs helper | `python/tvm/tirx/c6678_config.py` | ✅ |
| A.2 | Storage planning pass | `python/tvm/tirx/transform/c6678_storage_plan.py` | ✅（已挂入 `default_s_tir_pipeline`，但仅写 attrs，下游未消费） |
| A.3 | DMA legality / lowering | `python/tvm/tirx/transform/c6678_dma_legalize.py` | ✅（紧跟 A.9 `C6678DMALower` 之后挂入 `_c6678_post_schedule_passes`，同时校验 `load_row_major_tile` 与 `dma_trans`；幂等 `c6678.dma_legalized`；非 c6678 target 直接跳过；冒烟 9 用例 PASS；端到端 `via_build` 出码 2345 chars 不变） |
| A.4 | 算子特征抽取 pass | `python/tvm/tirx/analysis/c6678_features.py` | ✅（`extract_features(func, target)` 只读输出 `C6678PrimFuncFeatures`：op_kind / dom_kind / dom_extents / dtype / read_bufs / write_bufs / flop_count_static / tile_hint / static_alloc_l2_bytes；非 c6678 target 直接返回 None；冒烟 3 用例 PASS） |
| A.5 | 专家模板分发器 | `python/tvm/s_tir/dlight/c6678/dispatcher.py` | ✅（`ScheduleTemplate` 抽象 + `MatmulGemmTemplate` + `select_template(features, block_idx)`；`Matmul.apply` 已重构为薄壳：抽特征 → 选模板 → 模板就地改写 sch；自带 L2 容量门禁；冒烟 3 用例 PASS；端到端 `via_build` 零差分 2345 chars） |
| A.6 | 既有 `_p/_s` 模板 schedule 化 | `python/tvm/s_tir/dlight/c6678/{matmul,softmax,elementwise}.py` | 🟡（Matmul + Softmax phase A + ElementGreaterEqual same-shape 已接入；Matmul `_p_` 内层退化、softmax phase B、broadcast elementwise、LSTM 自动识别仍未完成） |
| A.7 | Bare-C entry signature（路径 2） | `python/tvm/tirx/transform/c6678_lower_entry.py`（新增）+ `python/tvm/s_tir/pipeline.py::_c6678_pre_packed_api_passes`（紧贴 `MakePackedAPI` 之前） | ✅（c6678 host PrimFunc 出 `void matmul_fp32(float* A, float* B, float* C)`，已绕过 FFI wrapper；通过 `calling_conv = kCPackedFunc` 让 `MakePackedAPI` 自动跳过） |
| A.8 | `C6678MulticoreLower`（路径 2） | `python/tvm/tirx/transform/c6678_multicore_lower.py` | ✅（识别 `ForKind.PARALLEL` → 改写成 `if (GetCoreNum > 0 && c6678_get_core_id >= 0) { for ... }` + 末尾 `C6678E_SyncN`，并给入口追加 `int32_t core_mask` 形参；与 A.7 链在 `_c6678_pre_packed_api_passes`） |
| A.9 | `C6678DMALower`（路径 2） | `python/tvm/tirx/transform/c6678_dma_lower.py` | ✅（拆成 `C6678DMALower` + `C6678AnnotateL2Alloc` 双 pass，pipeline 已挂入；demo `via_build` 出码 2345 chars，A/B 两段 DMA 第二参数均正确指向合并后 alloc 的对应段；若发现 `c6678.dma_load` 注解未被成功 lower，会直接报错而非静默跳过） |
| A.10 | `dlight.c6678.Matmul` 调度增强（路径 2） | `python/tvm/s_tir/dlight/c6678/matmul.py` | ✅（step1 已落地：`cache_read("global.l2") + compute_at(k_outer) + sch.annotate(c6678.dma_load/src_buffer/src_scope)`；A.10 step2 内层 `_p_` 退化开关待板上验证后再做） |

### 4.3 选址原则

- `tirx/transform/`：硬件相关、与 schedule 无关的 IR 改写 pass —— A.2 / A.3。
- `tirx/analysis/`：只读 IR 抽特征 pass —— A.4。
- `s_tir/dlight/c6678/`：调度模板，对齐 `s_tir/dlight/cpu|gpu/` 已有目录形态 —— A.5 / A.6。
- `python/tvm/tirx/c6678_config.py`：跨模块共享的 attrs 读取 / 默认值兜底 —— A.1，所有上面 pass 都依赖它。
- `python/tvm/contrib/c6678.py`：保持薄壳，仅 user-facing；底层 import 上述 helper，避免规则代码两份。

### 4.4 pass 注册风格（与现有 tirx 保持一致）

```python
@tvm.tirx.transform.prim_func_pass(opt_level=0, name="C6678StoragePlan")
class C6678StoragePlan:
    def transform_function(self, func, mod, ctx):
        target = func.attrs.get("target")
        if target is None or target.kind.name != "c6678":
            return func
        cfg = tvm.tirx.c6678_config.from_target(target)
        # 1. 遍历 buffer，按 scope 分配地址
        # 2. 写回 func.attrs["c6678.storage_plan"] 供下游 pass 使用
        return func
```

接入 pipeline 时，按 target 分支挂在 `s_tir/pipeline.py` 的 `default_s_tir_pipeline` 末尾、codegen 之前；
通过 `target.kind.name == "c6678"` 守卫，不影响其它 target。

### 4.5 与 `contrib/c6678.py` 的过渡关系

| 阶段 | `contrib/c6678.py` | `tirx`/`s_tir` |
|---|---|---|
| 当前 A.1+A.2 后 | 常量改为 re-export `tirx.c6678_config`；模板仍走字符串 | 多出 attrs helper + storage plan pass |
| A.3 后 | DMA 校验入口由 `validate_dma_path` 转调 pass | 多出 dma legalize pass |
| A.4 + A.5 后 | 模板拼字符串部分逐步切到 `tir.Schedule` | dlight/c6678 提供官方 schedule |
| A.6 完成后 | 退化成 `tvm.build(target='c6678')` 的便利封装 | 全链路 IR 化、可调优 |

### 4.6 解锁能力（动机回顾）

1. **特征分类驱动 schedule**：A.4→A.5 后，shape × dtype × scope 的不同组合可走不同 tile / cache_read 策略。
2. **DMA 安全性**：A.3 后，非法搬运（如 L1↔L2）会在 lowering 阶段直接报错，而非板卡跑挂。
3. **多核可视化**：A.2 + multicore lowering 后，每个 buffer 物理地址在 IR `attrs` 里可见。
4. **AutoTune 接入**：`s_tir/meta_schedule` 现成框架可挂上 c6678 schedule rule，从"专家模板"扩展到"专家模板 + 自动搜索"。

### 4.7 风险与权衡

- **Pass 不接入 pipeline ≠ 没用**：A.1/A.2 先以"独立可调用 + 单测覆盖"形态存在，**先打基础设施再接管线**，避免污染主流水线。
- **过渡期常量重复**：以 `tirx.c6678_config` 为单一事实源，`contrib/c6678.py` 仅做兼容性 re-export。
- **target.attrs Python 端不带 DefaultValue**：`Target("c6678").attrs` 在未显式赋值时**不会自动填入** `target_kind.cc` 的 `DefaultValue`，因此 `C6678Config.from_target` 在 Python 侧再做一次 `_C6678_DEFAULT_ATTRS` fallback。**任何下游 pass / schedule 必须通过 `from_target` 读取硬件常量**，不要直接读 `target.attrs`。
- **L2 是 per-core 资源**：A.2 在 plan 表里把 L2 buffer 的 `region_base/region_end` 暂填 `-1`，等 multicore lowering 拿到 `core_id` 再写真实地址。这意味着未来 `C6678MulticoreLower` 必须在 `C6678StoragePlan` 之后运行。
- **开发环境终端故障（2026-05-22 阻塞，已解除）**：曾经 `RunCommand` 在所有 zsh 终端上对任意命令均返回 exit 1 + 空 stdout/stderr，导致一段时间内无法跑 `/tmp/probe_a9_sblock.py` 实测 SBlock 结构、也无法跑 `Test4dsp/generate_c6678_matmul_via_build.py`。当前已恢复（2026-05-24 实测 `echo` / `python` / `bash` 等均能正常返回；conda env 名为 `tvm_env`，需先 `source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && conda activate tvm_env`）。教训：跑 demo / pytest 前必须 activate `tvm_env`，否则会因 `ModuleNotFoundError: No module named 'tvm'` 误判为代码问题。

### 4.8 路径 2 落地拆解（板上可运行形态）

> 触发原因：`tvm.tirx.build(target="c6678")` 当前输出 `int32_t __tvm_ffi_matmul_fp32(void*, void*, int32_t, void*)` 形态，BSP 端无 TVM runtime，无法链接。
> 对比 `Test4dsp/tests/generated_c6678_matmul.c`（旧字符串模板，板上可运行）：使用 `void fp_matmul_fusion_s(float *A, float *B, float *C, float *bias, const C6678MatmulConfig *cfg, int core_mask)` 这种 bare-C 入口，且通过 `GetLogicCoreId / C6678E_SyncN / dma_trans / DNUM` 等 BSP 调用承载多核与 DMA 语义，并使用 `LL2_*_BASE` 链接段宏定位 per-core L2 缓冲。
>
> 用户决定：**直接走路径 2（彻底端到端 IR 化），放弃路径 1（字符串模板托底）与路径 3（call_extern 注入模板函数）**，避免再受字符串模板的调优天花板。

#### 4.8.1 把目标 C 文件按段映射成 IR pass 责任

| `generated_c6678_matmul.c` 片段 | 责任 IR pass / 组件 | 现状 |
|---|---|---|
| `#include <78NE/initial.h>` 等 BSP 头 | `codegen_c6678.cc::CodeGenC6678::Init` | ✅ 已包含但当前调用图不引用 |
| `typedef struct { ... } C6678MatmulConfig` | A.10 `dlight.c6678.Matmul` 通过 `T.allocate` + `call_extern` 引出，或保留为 BSP 头中固定结构由 codegen 透出 | ⏳ |
| `void fp_matmul_fusion_p_(float *A, float *B, float *C, ...)` 内层核 | A.10 `dlight.c6678.Matmul`：当 tile 命中 `_p_` 形态时把核体降为 `tirx.call_extern` 或保留 IR 化的可调优内层 | ⏳ |
| `void fp_matmul_fusion_s(float *A, float *B, float *C, ..., int core_mask)` 外层 | A.7 bare-C entry + A.8 `C6678MulticoreLower` + A.9 `C6678DMALower` 联合产出 | 🟡 A.7 + A.8 已落地（入口已带 `int32_t core_mask`，外层多核分块 + `C6678E_SyncN` 已出；仍缺 `bias/cfg` 形参与 DMA 搬运） |
| `core_id = GetLogicCoreId(core_mask, DNUM)`、`C6678E_SyncN(core_num, core_id)` | A.8 `C6678MulticoreLower` 把 `parallel`/`thread_binding="coreIdx"` 转换为这套 BSP 调用 | ✅（IR 中以 `call_extern("c6678_get_core_id", core_mask)` 取 core_id（BSP 端封装 `GetLogicCoreId(core_mask, DNUM)` 隐藏 `DNUM` 宏），并以 `call_extern("C6678E_SyncN", ...)` 收尾；无需 C++ codegen 改动） |
| `ll2_A = (float *)LL2_A_BASE` | A.2 `C6678StoragePlan` + A.7 codegen：把 `c6678.storage_plan` 中 `scope == "global.l2"` 的 buffer 改写为 `LL2_*_BASE` 宏取址 | 🟡 plan attrs 已写，下游未消费 |
| `dma_trans / load_row_major_tile` | A.9 `C6678DMALower`：把 `cache_read scope="global.smc"`/`"global.l2"` 形态降级为 `tirx.call_extern("dma_trans", ...)`/`load_row_major_tile` | ⏳ |
| `int32_t main_func_handle(...)` 入口 | **不输出**：A.7 bypass `MakePackedAPI`，对 c6678 host func 直接发 bare 签名 | ✅（实现方式：c6678 host PrimFunc 在 `MakePackedAPI` 之前被 `C6678LowerEntry` 改写为 `params=[buf.data,...]` + `calling_conv=kCPackedFunc`，触发 `RequiresPackedAPI` 早返回） |

#### 4.8.2 子任务拆解（追加进 §4.2）

| 编号 | 名称 | 主要落点 | 依赖 |
|---|---|---|---|
| A.7 | Bare-C entry signature ✅ | **实际落点**：`python/tvm/tirx/transform/c6678_lower_entry.py`（pure Python `prim_func_pass`）+ 在 `python/tvm/s_tir/pipeline.py::default_s_tir_pipeline` 紧贴 `MakePackedAPI()` 之前插入 `_c6678_pre_packed_api_passes()`。**未改 C++**：核心利用 `src/tirx/transform/make_packed_api.cc::RequiresPackedAPI` 在 `calling_conv != kDefault` 时早返回；pass 把 PrimFunc 的 `params` 替换为各 `buffer.data` Var（type_annotation 已是 `T.handle("float32","global")`，CodeGenC 自动出 `float*`），`buffer_map={}`、`ret_type=TupleType([])`、`calling_conv=kCPackedFunc(1)`、`tirx.is_entry_func=True`。 | A.1 |
| A.8 | `C6678MulticoreLower` ✅ | **实际落点**：`python/tvm/tirx/transform/c6678_multicore_lower.py`（pure Python `prim_func_pass`）+ 在 `_c6678_pre_packed_api_passes()` 紧跟 A.7 之后。**未改 C++**：识别 `ForKind.PARALLEL` 循环（顶层 `parallel(extent)`），用 `tirx.stmt_functor.ir_transform` 把它替换为 `if (GetCoreNum > 0 && c6678_get_core_id >= 0) { for v=core_id*chunk; v < (core_id==N-1?extent:(core_id+1)*chunk); ++v) {<body>} }` + `Evaluate(call_extern("C6678E_SyncN", core_num, core_id))`，并在 `func.params` 末尾追加 `core_mask: int32`，落 `c6678.multicore_lowered=True` 标记保证幂等。**关键约束**：`tirx` 没有 `LetStmt`，因此 `core_num/core_id` 不能 let-binding 暂存，统一用闭包 `_core_num()/_core_id()` 在每个使用点都 inline `call_extern`（`GetCoreNum`、`c6678_get_core_id` 都是 BSP 端的纯函数，重复调用语义 OK）。 | A.2、A.7 |
| A.9 | `C6678DMALower` | `python/tvm/tirx/transform/c6678_dma_lower.py`：把 `cache_read scope="global.smc"/"global.l2"` 与 `c6678.storage_plan` 中的 region 信息合成 `dma_trans*`/`load_row_major_tile` 的 `tirx.call_extern`。**实测约束（探针 `/tmp/probe_a9_pipeline.py` + `/tmp/probe_a9_sblock.py`）**：① 单纯加 `sch.cache_read("global.l2")` + `sch.compute_at(..., k_outer)` 后，IR 在 `LowerOpaqueBlock + FlattenBuffer` 之后形态退化为 `for ax0,ax1 in T.grid(32,32): A_global_l2[ax0*32+ax1] = A[ax0_0*4096 + ax0*128 + ax2_0*32 + ax1]` —— `row0/col0/src_ld` 已被符号化进单一线性表达式。② 后续直接 build 会在 `LowerTVMBuiltin` 阶段触发 `Unknown device id in current IR`。③ 因此 A.9 必须在 schedule 端就给 staging block 注解 `c6678.dma_load = "load_row_major_tile"`，pass 在 `ConvertBlocksToOpaque` **之前**（`PlanAndUpdateBufferAllocationLocation` 之后）按 annotation 直接替换为 `Evaluate(call_extern("load_row_major_tile", ...))`。**当前实现（2026-05-24）**：拆成 `C6678DMALower`（DMA call 改写）+ `C6678AnnotateL2Alloc`（scope/annotation 兜底）两段；前者已能在 SBlock 阶段成功识别 `For(ax0)/For(ax1)/SBlockRealize(staging)` 三层结构并取出 `iter_values` 反向 substitute 出 `row0/col0`，后者负责在 `StorageRewrite` 之后给 `AllocBuffer(scope="global.l2")` 重建 `Buffer.data`（`PointerType.storage_scope = "global"`）+ 加 `disable_lower_builtin=True`。 | A.2、A.10 |
| A.10 | `dlight.c6678.Matmul` 调度增强 | 在 `s_tir/dlight/c6678/matmul.py` 里补 `cache_read("global.l2")` + `decompose_reduction` + 可选 `call_extern("fp_matmul_fusion_p_")` 的内层降级开关。**实测约束**：本步必须与 A.9 联合设计 —— 仅加 `cache_read+compute_at` 不打 annotation，IR 走到 `LowerTVMBuiltin` 会因 `global.l2` 取不到 device id 而崩；给 staging block 打 `c6678.dma_load` annotation 与 `row0_expr/col0_expr/rows/cols/src_ld` 元数据后，A.9 才有稳定输入。**当前实现（2026-05-24）**：step1 已落地 —— 在 `Matmul.apply` 的 `sch.reorder` 之后、`sch.parallel(i_outer)` 之前插入 `a_l2 = sch.cache_read(block, 0, "global.l2") + sch.compute_at(a_l2, k_outer, preserve_unit_loops=True) + sch.annotate(a_l2, "c6678.dma_load", "load_row_major_tile") + sch.annotate(a_l2, "c6678.src_buffer", "A") + sch.annotate(a_l2, "c6678.src_scope", "global")`，B 同形态；step2（`_p_` 内层 `call_extern` 退化开关）待 A.9 跑通端到端 demo 后再做。 | A.7 / A.8 / A.9 任一可解锁部分能力 |

> A.7 是必经之路：它把 codegen 出口从 "TVM runtime ABI" 切到 "BSP ABI"，之后 A.8 / A.9 / A.10 才能逐步把 IR 表达力填进去。

#### 4.8.3 推荐落地顺序与每一步的可观测产物

> 原则：**每一步都让 `Test4dsp/generate_c6678_matmul_via_build.py` 的输出更接近 `generated_c6678_matmul.c` 的可运行形态**，且每一步都跑通端到端 demo，不堆未验证改动。

1. **A.7（最高优先）** ✅：跳过 FFI wrapper。
   产物（实测）：`generated_c6678_matmul_via_build.c` 顶部从 `int32_t __tvm_ffi_matmul_fp32(void* self_handle, void* args, int32_t num_args, void* result)` 切到 `void matmul_fp32(float* A, float* B, float* C)`。via_build 输出体积从 17789 字符降至 1642 字符，baseline 从 17257 降至 1110。BSP 端可直接 `extern void matmul_fp32(...)` 链接，但仍是单核、纯 DDR、无 DMA。
   **回归**：`test_c6678_storage_plan.py` ✓、`probe_c6678_build_baseline.py` ✓、`test_c6678_matmul_codegen.py`（旧字符串模板路径）✓。
2. **A.8** ✅：`C6678MulticoreLower` 接管 `parallel` 注解。
   产物（实测）：`generated_c6678_matmul_via_build.c` 入口签名扩为 `void matmul_fp32(float* A, float* B, float* C, int32_t core_mask)`；外层 `T.parallel(4)` 已被替换为 `if ((GetCoreNum(core_mask) > 0) && (c6678_get_core_id(core_mask) >= 0)) { for (ax0_0 = core_id*chunk; ax0_0 < (core_id == N-1 ? 4 : (core_id+1)*chunk); ++ax0_0) {...} }`，函数尾部出 `C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask))`。via_build 输出体积从 1642 增至 2075 字符；baseline 仍为 1110（baseline schedule 中无 parallel 注解，A.8 自然不触发，符合幂等设计）。`c6678_get_core_id` 由 BSP 端封装 `GetLogicCoreId(core_mask, DNUM)`，IR 侧不感知 `DNUM` 宏。
   **回归**：`test_c6678_matmul_codegen.py`（DDR + L2）✓、`test_c6678_storage_plan.py` ✓、`probe_c6678_build_baseline.py` ✓。
3. **A.9 + A.10（必须联合 PR）**：`C6678DMALower` 接管 `cache_read`/`storage_plan` 中 L2 / SMC scope；`dlight.c6678.Matmul` 同步加 `cache_read("global.l2") + compute_at(k_outer)` 并对 staging block `sch.annotate(...)`。
   联合 PR 的硬约束（实测）：① 单加 A.10（cache_read+compute_at）会让 `LowerTVMBuiltin::VisitStmt_(AllocBufferNode*)` 触发 `Unknown device id in current IR` 直接报错；② 单加 A.9（无 schedule 端 annotation）则无稳定输入，pass 拿不到 `row0/col0/rows/cols/src_ld` 等必需字段（`LowerOpaqueBlock+FlattenBuffer` 后 staging copy 的索引已被符号化进单一线性表达式）。两步必须在同一变更里同时落地。
   落地 spec（A.10 schedule 端 / A.9 pass 端契约）：
   - schedule 端：`A_l2 = sch.cache_read(block, 0, "global.l2")`、`B_l2 = sch.cache_read(block, 1, "global.l2")`，`sch.compute_at(A_l2, k_outer, preserve_unit_loops=True)` 同上 B；随后 `sch.annotate(A_l2, "c6678.dma_load", "load_row_major_tile")` + `sch.annotate(A_l2, "c6678.src_buffer", "A")` + `sch.annotate(A_l2, "c6678.src_scope", "global")`，B 同理。
   - A.9 pass 端：拆为两段。① `C6678DMALower` 在 `ConvertBlocksToOpaque` **之前**（`PlanAndUpdateBufferAllocationLocation` 之后）执行，要求 `SBlockRealize.iter_values` 仍然存在；匹配 `For(ax0)/For(ax1)/SBlockRealize(staging_block)` 三层结构，取 `staging_realize.iter_values` 把 `ax0/ax1` 替换为 0 得到 `row0/col0`、外层 `For.extent` 即 `rows/cols`、`block.reads[0].buffer.shape[-1]` 即 `src_ld`，把整个三层 For/SBlockRealize 替换为 `Evaluate(call_extern("", "load_row_major_tile", src.data, dst.data, row0, col0, rows, cols, src_ld, IntImm("int32", elem_bytes), StringImm(src_scope)))`。② `C6678AnnotateL2Alloc` 在 `StorageRewrite` **之后**、`LowerTVMBuiltin` **之前**执行（`StorageRewrite` 会丢弃 `AllocBuffer` 上除 `kVolatile` 外的所有 annotation，所以必须在它之后跑）；该 pass 找到所有 `buffer.scope() == "global.l2"` 的 `AllocBuffer`，重建 `Buffer.data` Var（`PointerType.storage_scope = "global"`）并通过 `stmt_functor.substitute` 把 body 中所有引用一并替换，并追加 `disable_lower_builtin = True` 兜底注解（背景见 `src/tirx/transform/lower_tvm_builtin.cc::VisitStmt_(AllocBufferNode)` 第 244-259 行 + `src/target/source/codegen_c.cc::PrintStorageScope` 第 434-436 行 `TVM_FFI_ICHECK_EQ(scope, "global")`）。
   - Pipeline 接入：新增 `_c6678_post_schedule_passes()` 返回 `[tirx.transform.C6678DMALower()]`，挂在 `s_tir.transform.PlanAndUpdateBufferAllocationLocation()` 之后、`s_tir.transform.ConvertBlocksToOpaque()` **之前**；`tirx.transform.C6678AnnotateL2Alloc()` 单独挂在 `tirx.transform.StorageRewrite()` 之后；其它 target 不受影响（pass 内部对 `target.kind.name == "c6678"` 做守卫）。
   产物（目标）：`generated_c6678_matmul_via_build.c` 出现 `float A_global_l2[1024];` + `load_row_major_tile(A, A_global_l2, ax0_0*32, ax2_0*32, 32, 32, 128, 4, "global");` + 对应 B 同形态调用，与 `generated_c6678_matmul.c` 的 fusion_s 外层一致。via_build 输出体积预计从 2075 增至 2800~3500 字符。
   **当前实现状态（2026-05-24，✅ 已落地）**：
   - 文件落点：[`c6678_dma_lower.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py)（双 pass，约 280 行）；[`__init__.py`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/__init__.py) 增加 `C6678DMALower, C6678AnnotateL2Alloc` 导出；[`pipeline.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py) 增加 `_c6678_post_schedule_passes()` 挂载点 + 在 `StorageRewrite` 之后挂 `C6678AnnotateL2Alloc`；[`s_tir/dlight/c6678/matmul.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/matmul.py) 在 `sch.reorder` 之后、`sch.parallel(i_outer)` 之前注入 `cache_read + compute_at + annotate`。
   - 已验证：`C6678DMALower` 部分按 `C6678_DMA_DEBUG=1` 加诊断打印实测：`has_dma_annotation=True`、`func.body type=SBlockRealize`、`ir_transform` 在 SBlock 阶段确实命中两段 staging block；`C6678AnnotateL2Alloc` 实测 `AllocBuffer total=1, global.l2=1, BufferLoad rewrites=2, BufferStore rewrites=0, var_remap=1`（`StorageRewrite` 之后两段 staging buffer 已合并为一段，符合预期）。`_annotate_l2_alloc` 关键设计：第一轮 `ir_transform` 通过 `tirx.decl_buffer(scope="global", data=new_data)` 重建 `Buffer.data` 的 `PointerType.storage_scope`；第二轮 `ir_transform` 把所有 `BufferLoad/BufferStore` 的 `buffer` 字段以 `node.buffer.data` 为索引替换为新 Buffer；最后 `stmt_functor.substitute` 把 PrimExpr 里的 data Var 引用一并替换。
   - 实测产物：`generated_c6678_matmul_via_build.c` 出码 2345 chars（A.8 状态 2075，A.9+A.10 step1 后 2309，A.9 follow-up 修复后 2345），baseline 仍 1110 chars 不变；3 个回归测试 `test_c6678_storage_plan / probe_c6678_build_baseline / test_c6678_matmul_codegen` 全部 PASS。
   - **Follow-up 已闭环（2026-05-24）**：`StorageRewrite` 把 `A_global.l2` 与 `B_global.l2` 合并到同一块 alloc + 用 elem_offset=16384 区分（出码层面体现为 `float A_global_l2[32768];` 单一段）。原本 `call_extern("load_row_major_tile", ..., dst.data, ...)` 是不透明节点，`StorageRewrite` 不会改写其参数列表，导致 B 的 DMA 第二参数仍是 `A_global_l2` 头部。**采纳方案 (a)**：在 [`C6678DMALower::_build_dma_call`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py) 用 `src_buf.access_ptr("r") / dst_buf.access_ptr("w")` 替代裸 `.data`，`tvm_access_ptr` builtin 会被 [`storage_rewrite.cc:1601-1625`](file:///home/tangqingyun/tvm/src/tirx/transform/storage_rewrite.cc#L1601-L1625) 专门识别并合成正确偏移参数；后续 `LowerIntrin` 将 `tvm_access_ptr` 降级为 `address_of(BufferLoad)`，C codegen 出 `&A_global_l2[16384]` 形态。修复后实测出码：
     ```c
     load_row_major_tile((&(A[0])), (&(A_global_l2[0])),     (ax0_0 * 32), cse_v1, 32, 32, 128, 4, "global");
     load_row_major_tile((&(B[0])), (&(A_global_l2[16384])),  cse_v1, cse_v2, 32, 32, 128, 4, "global");
     ```
     方案 (b)（`C6678DisableMergeForL2` 阻止合并）保留作为未来 SMC scope 等场景的备选。
4. **A.10 自动调优分支（A.9 落地之后再做）**：`dlight.c6678.Matmul` 给 `_p_` 内层加 `call_extern("fp_matmul_fusion_p_", ...)` 退化开关；同时保留 IR 化内层用于自动调优搜索分支。
   产物：当 schedule 命中 `_p_` 模板时，输出代码与现有 `generated_c6678_matmul.c` 完全等价；而走自动调优分支时，可以让 IR 化内层接 MetaSchedule。

#### 4.8.4 决策与放弃理由（路径 1 / 3）

| 路径 | 优点 | 放弃理由 |
|---|---|---|
| 路径 1：旧字符串模板托底 + 新流水线只跑特征/分发 | 板上立刻可运行 | 调优天花板锁死在模板字符串拼接；与 §4.1 终极形态目标背离 |
| 路径 3：保留 FFI wrapper，但对内层 `call_extern` 注入旧模板函数 | 改动最少 | runtime 必须有 TVM；BSP 端无法链接，仍解决不了首要 gap |
| **路径 2：彻底 IR 化**（已选；A.7 + A.8 已落地） | 终态符合 §4.1；调优能力贯通；`tvm.build(mod, target="c6678")` 一行打穿 | 需要新增 A.7~A.10 共 4 个组件；A.7 + A.8 已完成（FFI wrapper 已脱、多核派发 + SyncN 已在 IR 层产出），A.9 / A.10 落定前 demo 暂不能直接上板（仍缺 DMA + L2 staging） |

### 4.9 PR-S2 phase A 关键发现（2026-05-25）

PR-S2 phase A（softmax 端到端 IR 化）落地过程中踩到两个非平凡的坑，已固化成可复用模式，**任何后续新算子（conv / reduce / layernorm / ...）规划 dlight 模板时都必须参考**。

#### 4.9.1 PrimFunc 中间 buffer 必须用 `T.sblock_alloc_buffer`，不能用 `T.alloc_buffer`

| 维度 | `T.sblock_alloc_buffer((shape,), "dtype")` | `T.alloc_buffer((shape,), "dtype")` |
|---|---|---|
| IR 层语义 | 把 buffer 加到最近 `SBlock`（或 root PrimFunc）的 `alloc_buffers` 列表 | 生成独立 `AllocBuffer` 语句节点 |
| 与 `compute_at` 的关系 | ✅ schedule 端 `IsOutputBlock` 会在 `scope_root.alloc_buffers` 中找到，把它判为"中间 buffer" | ❌ schedule 端 `IsOutputBlock` 在 `alloc_buffers` 中找不到，把写入它的 block 误判为 output → `ScheduleError: ... is an output block` |
| 与 topi 一致性 | ✅ topi.nn.softmax / topi.nn.layer_norm 等都走这条路径（IR 文本头部有 `# with T.sblock("root")` + `T.sblock_alloc_buffer(...)`） | ⚠️ 仅在 PrimFunc 没有任何 schedule 改写需求时可用 |

**判断准则**：只要后续会被 `compute_at` 移到外层循环里（即"作为中间 staging"使用），**必须**用 `sblock_alloc_buffer`。源码佐证：[`s_tir/schedule/analysis/analysis.cc::IsOutputBlock`](file:///home/tangqingyun/tvm/src/s_tir/schedule/analysis/analysis.cc) 检查 `scope_root->alloc_buffers` 集合，[`compute_at.cc::CheckNotOutputBlock`](file:///home/tangqingyun/tvm/src/s_tir/schedule/primitive/compute_at.cc) 在 `ComputeAtOrReverseComputeAtImpl` 起始处调用。

实战示例：[`Test4dsp/generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py)（已加注释说明）。

#### 4.9.2 c6678 上 `scope='global'` 大 alloc 必须打 `disable_lower_builtin = True`

**症状**：在 c6678 上 `tvm.tirx.build` 一份带中间 buffer 的 PrimFunc（例如 softmax `T_exp[8, 1024]` = 32KB），`LowerTVMBuiltin` 阶段崩溃：

```
tvm.error.InternalError: Check failed: (device_id_) is false: Unknown device id in current IR
  at src/tirx/transform/lower_tvm_builtin.cc::VisitStmt_(AllocBufferNode*)
```

**根因链**（参考 [`lower_tvm_builtin.cc:240-265`](file:///home/tangqingyun/tvm/src/tirx/transform/lower_tvm_builtin.cc)）：
1. `kMaxStackAlloca = 1024` ([`device_api.h:119`](file:///home/tangqingyun/tvm/include/tvm/runtime/device_api.h#L119))；
2. 当 `constant_size * elem_bytes > kMaxStackAlloca` 时（softmax 32KB > 1024）走 fast path 失败，退化为 `TVMBackendAllocWorkspace(device_type, device_id, nbytes, ...)`；
3. workspace 路径需要 `device_id_` AttrStmt 在 IR 中提前出现，但 c6678 是 `kDLCPU` bare-metal，没有 device_id 概念，导致 ICHECK 触发。

**解药**：[`lower_tvm_builtin.cc:244-248`](file:///home/tangqingyun/tvm/src/tirx/transform/lower_tvm_builtin.cc#L244-L248) 提供了一条逃生通道 —— `AllocBufferNode.annotations["disable_lower_builtin"] == True` 时直接 `return stmt;`（保留为栈 alloca，由 CodeGenC 直出 `float vid[N];`）。

**已固化的两层防护**（按 pipeline 顺序）：

| pass | 处理范围 | 文件 |
|---|---|---|
| `C6678AnnotateL2Alloc` | `scope == "global.l2"` 的 alloc（A.9 历史落地） | [`c6678_dma_lower.py:395`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L395) |
| **`C6678AnnotateGlobalAlloc`**（PR-S2 phase A 新增） | `scope == "global"` 的 alloc（包含 A.9 已 rewrite 过的 + softmax 等带中间 buffer 的算子原生 `global` alloc） | [`c6678_dma_lower.py:467`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) |

挂载顺序：[`pipeline.py`](file:///home/tangqingyun/tvm/python/tvm/s_tir/pipeline.py) `StorageRewrite` 之后先跑 `C6678AnnotateL2Alloc`（把 `global.l2 → global`），再跑 `C6678AnnotateGlobalAlloc`（覆盖所有 `global` alloc，包含上一步刚改写的 + 算子原生的）。两个 pass 都对 `target.kind.name == "c6678"` 守卫，幂等（`c6678.l2_alloc_annotated` / `c6678.global_alloc_annotated` flag）。

**对后续算子的影响**：matmul 没有非 staging 中间 alloc，A.9 单独已经够；softmax / layernorm / RMSNorm / reduce 之类带 `T_max / T_sum / T_exp / T_var` 中间 buffer 的算子，必须依赖 `C6678AnnotateGlobalAlloc`。它已挂在主 pipeline，**算子作者无需手动注解**。

#### 4.9.3 端到端验证产物（基线快照）

| 算子 | demo | scheduled 出码 | baseline 出码 | 形式化回归 |
|---|---|---|---|---|
| matmul (fp32 128×128×128) | [`generate_c6678_matmul_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_matmul_via_build.py) | **2345 chars** | 1110 chars | [`test_c6678_end_to_end.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_end_to_end.py)（7/7） |
| **softmax (fp32 8×1024)** | [`generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) | **1925 chars** | 1641 chars | [`test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py)（7/7） |

softmax 出码关键特征（与 §4.1.3 BSP ABI 契约一致，仅多 C99 标准库符号 `expf`）：

```c
void softmax_fp32(float* A, float* Out, int32_t core_mask);

void softmax_fp32(float* A, float* Out, int32_t core_mask) {
  if ((GetCoreNum(core_mask) > 0) && (c6678_get_core_id(core_mask) >= 0)) {
    int32_t ax0;
    for (ax0 = (c6678_get_core_id(core_mask) * (8 / GetCoreNum(core_mask))); ...) {
      float T_max[1];      /* sblock_alloc_buffer((1,))，命中 kMaxStackAlloca fast path */
      float T_exp[1024];   /* sblock_alloc_buffer((INNER,)) = 4KB > kMaxStackAlloca，
                              依赖 disable_lower_builtin 走栈 alloca */
      float T_sum[1];
      for (ax1 = 0; ax1 < 1024; ++ax1)   { /* pass 1: max */ }
      for (ax1_1 = 0; ax1_1 < 1024; ++ax1_1) {
        T_exp[ax1_1] = expf((A[(cse_v1 + ax1_1)] - T_max[0]));   /* pass 2: exp */
      }
      for (ax1_2 = 0; ax1_2 < 1024; ++ax1_2) { /* pass 3: sum */ }
      for (ax1_3 = 0; ax1_3 < 1024; ++ax1_3) { /* pass 4: div */ }
    }
  }
  C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask));
}
```

注：`compute_at(outer, preserve_unit_loops=True)` 后，`T_exp` 的内部循环 `ax1_1` 仍是 `INNER=1024` 长度（因为 axis 是 reduce 轴，不会被 outer 吸收）；外层 `ax0` 由多核派发拆分。

---


## 5. 接手索引

接手时按以下顺序使用本仓库的文档：

| 想了解 | 去哪里 |
|---|---|
| 项目背景、整体架构、硬件假设 | 本文 §1~§2 |
| 当前能力到哪一步、还缺什么 | 本文 §3 |
| 下一步该做什么、为何这样设计 | 本文 §4 |
| 某个函数/pass 的 API、参数、返回值 | [`api_reference.md`](./api_reference.md) |
| target attrs 完整表 | [`api_reference.md` §1](./api_reference.md#1-c6678-target-属性表) |
| `C6678Config` / `from_target` | [`api_reference.md` §2](./api_reference.md#2-tvmtirxc6678_config) |
| `C6678StoragePlan` 的 schema | [`api_reference.md` §3](./api_reference.md#3-tvmtirxtransformc6678storageplan) |
| `build_matmul_module` 全部参数 | [`api_reference.md` §4](./api_reference.md#4-tvmcontribc6678) |
| 命令白名单（conda + PYTHONPATH） | [`dev_history.md` §1](./dev_history.md) |
| SOP（怎么开发、怎么验证） | [`dev_history.md` §2](./dev_history.md) |
| 调用示例 / 已知问题 | [`dev_history.md` §3~§4](./dev_history.md) |
| MVP 探索史与算子选择规则细节 | [`dev_history.md` §5](./dev_history.md) |
| 历次变更摘要 / 历史阶段计划 | [`dev_history.md` §6~§7](./dev_history.md) |

### 关键源文件清单

| 文件 | 角色 |
|---|---|
| `src/target/target_kind.cc` | c6678 TargetKind + 14 项 hardware attrs |
| `src/target/source/codegen_c6678.cc` | `target.build.c6678` 注册 |
| `python/tvm/tirx/build.py` | c6678 host kind 路由 |
| `python/tvm/tirx/c6678_config.py` | **硬件常量唯一事实源** |
| `python/tvm/tirx/transform/c6678_storage_plan.py` | storage planning pass（A.2） |
| `python/tvm/s_tir/pipeline.py` | `_c6678_specific_passes()`：c6678 专属 pass 在 `default_s_tir_pipeline` 里的接入点 |
| `python/tvm/s_tir/dlight/c6678/{__init__,base,matmul}.py` | 极简 matmul schedule（A.5/A.6 起步） |
| `python/tvm/contrib/c6678.py` | matmul MVP 入口（薄壳，老路径） |
| `Test4dsp/tests/test_c6678_storage_plan.py` | A.2 冒烟测试 |
| `Test4dsp/tests/test_c6678_matmul_codegen.py` | matmul codegen 测试（老路径） |
| `Test4dsp/tests/probe_c6678_build_baseline.py` | 端到端最小调用探测（含 A.2 attrs 可视化） |
| `Test4dsp/generate_c6678_matmul.py` | 老路径生成示例（字符串模板） |
| `Test4dsp/generate_c6678_matmul_via_build.py` | **新路径端到端 demo**：`tvm.tirx.build(target="c6678")` |

---

## 6. 一句话总结

当前项目已经完成了 **6678DSP 的 matmul MVP 源码生成路径** 与 **A.1 / A.2 / A.3 / A.4 / A.5 / A.6(matmul + softmax + ElementGreaterEqual S3 + LSTM extern wrapper) / A.7 / A.8 / A.9 / A.10 step1 + PR-S2 phase A**：

- 硬件常量在 `tirx.c6678_config` 集中管理；
- `C6678StoragePlan` 已挂入 `default_s_tir_pipeline`（仅写 attrs，下游不消费）；
- **A.4 已落地**：`python/tvm/tirx/analysis/c6678_features.py` 提供只读 `extract_features(func, target)` API，输出 `C6678PrimFuncFeatures(blocks=tuple[C6678OpFeatures], config=C6678Config)` 三元 frozen dataclass（per-block 给 op_kind / dom_kind / dom_extents / dtype / read_bufs / write_bufs / flop_count_static / tile_hint / static_alloc_l2_bytes），non-c6678 target 直接返回 None；不挂 pipeline，由 A.5 dispatcher 直接消费。
- **A.5 已落地**：`python/tvm/s_tir/dlight/c6678/dispatcher.py` 提供 `ScheduleTemplate` 抽象 + `MatmulGemmTemplate` + `select_template(features, block_idx)`；`Matmul.apply` 退化为薄壳（抽特征 → 选模板 → 模板就地改写 sch）；派发器自带 L2 容量门禁（按 `feats.config.l2_size` 卡更保守的读 buffer staging 全量 `static_alloc_l2_bytes`；128×128 matmul 为 131072B，与实际 `float A_global_l2[32768]` 对齐）；零差分验证：`via_build` 出码仍为 2345 chars，3 个 dispatcher 冒烟用例（gemm 命中 / elementwise None / 容量门禁拦超量模板）全部 PASS。
- **A.3 已落地**：`python/tvm/tirx/transform/c6678_dma_legalize.py` 紧跟 A.9 `C6678DMALower` 之后挂入 `_c6678_post_schedule_passes`，对 `call_extern("load_row_major_tile", ...)` 与 `call_extern("dma_trans", ...)` 做纯静态校验（scope 白名单 / 正数性 / 单次搬运不超过 `dma_max_transfer` / `cols*elem_size` 非 `dma_align_bytes` 整数倍只发 `C6678DMAAlignmentWarning` 不阻断）；只读不改 IR，幂等 `c6678.dma_legalized` flag，非 c6678 target 直接跳过；冒烟 9 用例（合法 / 非法 scope / 超容量 / 行对齐告警 / 幂等 / 非 c6678 跳过）全部 PASS；端到端 `via_build` 出码仍为 2345 chars。
- **A.7 已落地**：`C6678LowerEntry`（pure Python pass）通过把 PrimFunc `params` 替换为各 `buffer.data` Var 并显式置 `calling_conv = kCPackedFunc`，使 `MakePackedAPI` 自动跳过 c6678 host func，输出由 `int32_t __tvm_ffi_matmul_fp32(void*, void*, int32_t, void*)` 直接降为 bare-C 签名；
- **A.8 已落地**：`C6678MulticoreLower`（pure Python pass）识别 `ForKind.PARALLEL` 顶层并行循环，在 IR 层用 `tirx.stmt_functor.ir_transform` 把它降级为 `if (GetCoreNum > 0 && c6678_get_core_id >= 0) { for ... }` + 末尾 `C6678E_SyncN`，并在入口形参末尾追加 `int32_t core_mask`；与 A.7 同处 `_c6678_pre_packed_api_passes()` 紧贴 `MakePackedAPI` 之前；
- **A.9 + A.10 step1 已落地**：dlight 端 `cache_read("global.l2") + compute_at(k_outer) + sch.annotate("c6678.dma_load", ...)`（A.5 落地后改由 `MatmulGemmTemplate.apply` 完成）；pipeline 端拆成 `C6678DMALower`（替换 staging block 为 `Evaluate(call_extern("load_row_major_tile", src.access_ptr("r"), dst.access_ptr("w"), ...))`）+ `C6678AnnotateL2Alloc`（`scope="global.l2"` 的 AllocBuffer 重建为 `scope="global"`）两段；实测 `via_build` 出码 2345 chars，A 段 `(&(A[0]))/(&(A_global_l2[0]))`、B 段 `(&(B[0]))/(&(A_global_l2[16384]))`；
- **PR-S2 phase A 已落地（2026-05-25）**：dlight 端新增 [`Softmax`](file:///home/tangqingyun/tvm/python/tvm/s_tir/dlight/c6678/softmax.py) rule（4 块 PrimFunc：max → exp → sum → div；调度顺序 `parallel(outer)` 在前、`for bi in reversed(block_infos[:-1]): compute_at(bi.block_rv, outer, preserve_unit_loops=True)` 在后，与 `dlight/cpu/reduction.py` 一致）；pipeline 端新增 [`C6678AnnotateGlobalAlloc`](file:///home/tangqingyun/tvm/python/tvm/tirx/transform/c6678_dma_lower.py#L467) 紧跟 `C6678AnnotateL2Alloc` 之后，给所有 `scope='global'` 的 AllocBuffer 自动追加 `disable_lower_builtin=True`，避开 c6678 bare-metal 没有 `device_id` 时 `LowerTVMBuiltin` 的崩溃路径；端到端 demo [`generate_c6678_softmax_via_build.py`](file:///home/tangqingyun/tvm/Test4dsp/generate_c6678_softmax_via_build.py) 实测 1925 chars 出码（含 `expf(...)` 直降、`T_max[1]/T_exp[1024]/T_sum[1]` 栈数组、4 串行内层、多核派发 + `C6678E_SyncN`），matmul 端零差分（仍 2345 chars）；形式化回归 [`test_c6678_softmax_codegen.py`](file:///home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py) 7/7 PASS；两条核心教训（`sblock_alloc_buffer` vs `alloc_buffer`、`disable_lower_builtin` 注解防护）已固化在 §4.9。
- `tvm.tirx.build(mod, target="c6678")` 一行端到端跑通 fp32 matmul + softmax + same-shape ElementGreaterEqual → bare-C 源码 + 多核派发 BSP 调用；matmul 已有 L2 staging + DMA 搬运，softmax 暂未做 DMA staging；ElementGreaterEqual 已完成输入侧 dma_trans staging + 1D L2 compact；BSP 端可 `extern void matmul_fp32(float*, float*, float*, int32_t)` / `extern void softmax_fp32(float*, float*, int32_t)` / `extern void greater_equal_fp32(float*, float*, int8_t*, int32_t)` 直接链接；
- `contrib.c6678` 退到薄壳，仅 re-export 常量、生成字符串模板，作为旧路径并存。

下一步按 §4.8 路径 2 推进：**PR-S2 phase B（softmax + dma_trans staging：BLOCK_CAPACITY=115200/32768 分块 + `cache_read("global.l2")` + `sch.annotate("c6678.dma_load", "dma_trans")`）** 与 **A.10 step2（_p_ 内层退化开关，需先解决 staging stride 与 BSP `_p_` 签名不匹配问题）** 并行收尾 —— PR-S2 phase A 已于 2026-05-25 落地，对齐 [`examples.md`](./examples.md) `fp_softmax_p` 的板上可运行形态（仅缺 staging）。


### 4.10 2026-05-30 评审后修正记录（中期展示前）

本轮根据代码评审结论完成了几项务实修正，目的是减少“能出码但隐藏风险”的问题：

1. **L2 容量估算修正**：`C6678Features.static_alloc_l2_bytes` 不再只估算两个 32×32 tile（8192B），而是按当前 matmul schedule 最终可能生成的 A/B 两个读 buffer staging 全量估算；128×128×128 fp32 matmul 现在为 131072B，与实际 C 源码中的 `float A_global_l2[32768]` 对齐。这样 dispatcher 的 L2 gate 更保守，不会用过小估算误放候选模板。
2. **DMA lower 失败显式化**：`C6678DMALower` 在执行后会检查是否还残留 `c6678.dma_load` 注解；若残留，说明某个 staging block 未被成功改写为 `load_row_major_tile` 或 `dma_trans`，现在会直接抛错，不再静默标记 `c6678.dma_lowered=True`。
3. **移除 codegen 实验硬编码**：删除 `CodeGenC6678` 中把 `__tvm_ffi_main` 强制改名为 `test_func` 的逻辑，避免未来用户符号或多入口场景被实验代码污染。
4. **修复 `test_my_ops.py`**：测试脚本改为自动编译 `Test4dsp/src/total_test.c` 到 `Test4dsp/build/libtotal_test.so`，再用 `tvm_ffi.load_module()` 加载并调用 `mod.add_one_c(x, y)`；不再使用过期的 `libinfo.load_lib_ctypes("libtotal_test.so")` 调用方式。

已验证命令：`test_c6678_features.py`、`test_c6678_dispatcher.py`、`test_c6678_end_to_end.py`、`test_my_ops.py` 均在 `tvm_env` 下通过。LLVM target canonicalizer warning 属于当前 TVM 构建未启用 LLVM 的环境噪声，不影响上述 c6678 路径验证。

ElementGreaterEqual 已完成第三阶段的输入 DMA staging + 1D L2 compact：新增 `dlight.c6678.ElementGreaterEqual`，支持 same-shape `float32 >= float32 -> bool` 的单 block TVMScript，按 L2 容量 helper 切分外层，两个输入经 `cache_read("global.l2")` 和 `c6678.dma_load="dma_trans"` 进入 `C6678DMALower`；pass 在 TIR 层生成 tile-dependent `dma_trans((&A[offset]), &L2[0], size_bytes)` / `dma_trans((&B[offset]), &L2[tile], size_bytes)`，并把后续 L2 load/store 改写为 tile-local index。当前 bool storage 在 C codegen 中落为 `int8_t*`，与 `AGENTS.md` 的 `bool*` BSP 参考实现存在 ABI 表述差异，后续需要统一。broadcast、输出 DMA store、直接调用 `fp_greater_equal_s` 的 extern 退化开关尚未实现。


### 4.11 2026-05-30 ElementGreaterEqual 与 LSTM 接入记录

本轮在不夸大完成度的前提下，补了两个面向中期展示更有代表性的算子入口：

1. **ElementGreaterEqual（EGE-S3）**：`python/tvm/s_tir/dlight/c6678/elementwise.py` 已导出 `ElementGreaterEqual` schedule rule。当前匹配一个无 reduction 的 injective block，要求 block body 为 `Out[...] = lhs >= rhs`（IR 中是 `tirx.GE`），按 `l2_dma_block_elems` 切分外层，并对 A/B 两个输入插入 `cache_read("global.l2") + c6678.dma_load="dma_trans"`，复用 A.8 `C6678MulticoreLower` 生成 `core_mask` 多核派发。回归 `test_c6678_greater_equal_codegen.py` 已通过，生成入口形如 `void greater_equal_fp32(float* A, float* B, int8_t* Out, int32_t core_mask)`，源码包含 tile-dependent `dma_trans` 和 compact L2 staging（当前回归中合并后为 `float A_global_l2[174080]`，不再是全量 `524288`）。注意：这是 same-shape 输入 DMA staging + 1D L2 compact MVP，尚未支持 `AGENTS.md` 参考实现里的 scalar/broadcast 分支，也未做输出 DMA store。
2. **LSTM extern wrapper（LSTM-S0）**：新增 `test_c6678_lstm_extern_codegen.py`，证明 TVMScript 可以生成一个 bare-C wrapper，直接调用 BSP 侧复合核 `fp_lstm_s(Output, Input, Params, core_mask)`。这一步把 LSTM 纳入 TVM/c6678 出码链路，但不是完整的 LSTM pattern matcher；自动识别 gate matmul、sigmoid/tanh、zoneout、projection、bidirectional 等子图仍是后续工作。

新的定向验证命令已加入 `dev_history.md`。截至本记录，matmul、softmax phase A、ElementGreaterEqual S3、LSTM extern wrapper、DMA lower failure、FFI demo 均可在 `tvm_env` 下跑通。


### 4.12 2026-05-30 ElementGreaterEqual 输入 DMA staging + 1D L2 compact

本轮开始实现 L2/DMA 分块逻辑，先选择最简单、可验证的 ElementGreaterEqual same-shape 路径，而不是直接改 softmax reduction 或 LSTM 复合核。

已完成：

1. `python/tvm/tirx/c6678_config.py` 新增 `dtype_nbytes`、`align_down`、`l2_dma_block_elems`，用于按 `l2_size / dma_align_bytes / reserve_bytes / staging buffer 数量` 计算 1D DMA tile 容量。
2. `dlight.c6678.ElementGreaterEqual` 从单纯 `parallel(outer)` 升级为：按 L2 tile capacity split 外层，A/B 两个输入分别 `cache_read("global.l2")`，并标注 `c6678.dma_load="dma_trans"`。
3. `C6678DMALower` 的 `dma_trans` lowering 修正了 tile offset：不再每个 tile 都从 `A[0]/B[0]` 搬运，而是从 staging block 的 `iter_values` 计算 `start_expr`，生成 `&A[offset] / &B[offset]`；tail tile 的 `size_bytes` 使用 `min(extent, shape - offset) * elem_bytes`。
4. `C6678DMALower` 内部新增 1D compact remap：为每个 `dma_trans` staging 记录 old buffer、compact buffer、tile start、tile extent、valid extent；DMA 目标指针改为 compact L2 offset 0；后续 L2 `BufferLoad/BufferStore` 改写为 `index - tile_start`。
5. `test_c6678_greater_equal_codegen.py` 已扩大到 `N=262144`，确保出现多个 tile 和多核派发，并断言源码中存在 `dma_trans`、tile-dependent 输入指针、compact L2 staging storage、tile-local L2 index、`GetCoreNum` 和 `C6678E_SyncN`。当前生成形态中，StorageRewrite 会把 A/B 两个 87040 元素的 L2 tile 合并为 `float A_global_l2[174080]`，不再生成 `float A_global_l2[524288]`。

仍未完成且需要诚实说明：

- 当前 compact 仍是 TVM 栈数组风格的 L2 staging，还不是 `examples.md` 中 `get_l2_addr(core_id)` 形式的 per-core L2 base 指针。后续如果要更贴近 BSP 模板，需要新增 pointer-style L2 region planning。
- 输出仍直接写 DDR，没有实现 `cache_write("global.l2") + dma_store`。
- broadcast/scalar GreaterEqual 仍未接入，`AGENTS.md` 中 `fp_greater_equal_s` 的完整 BSP 分支尚未作为 extern fallback 使用。
