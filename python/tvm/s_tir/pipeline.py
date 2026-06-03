# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# pylint: disable=invalid-name
"""The S-TIR backend compilation pipeline."""

import tvm
from tvm import s_tir, tirx
from tvm.tirx import pipeline as tir_pipeline


def _c6678_specific_passes():
    """C6678 专属的前置 pass 序列（流水线起点）。

    目前只包含 A.2（`C6678StoragePlan`，识别但不改写），后续 A.3 / multicore lower
    等会按顺序追加在这里。所有 pass 自身都对 `target.kind.name == "c6678"` 做守卫，
    其它 target 走到这里时直接原样返回，不会污染主流水线。

    注：A.7 ``C6678LowerEntry`` 改写的是 PrimFunc 入口签名（params / buffer_map /
    calling_conv），必须在 ``MakePackedAPI`` 之前的最后一刻执行，因此它单独通过
    ``_c6678_pre_packed_api_passes`` 挂载，而不是放在本函数里。
    """
    return [tirx.transform.C6678StoragePlan()]


def _c6678_post_schedule_passes():
    """紧贴 ``LowerOpaqueBlock`` 之前执行的 c6678 专属 pass 序列。

    顺序约束：必须**在 ``LowerOpaqueBlock`` 之前**，因为 A.9 ``C6678DMALower``
    需要消费 SBlock 阶段的 ``annotations``、``BufferRegion``、``SBlockRealize.iter_values``
    等结构化信息——一旦 ``LowerOpaqueBlock`` + ``FlattenBuffer`` 完成，
    staging copy 的索引会被符号化进单一线性表达式，``row0_expr/col0_expr/rows/cols/src_ld``
    将无法直接还原。

    并发约束：必须**在 ``PlanAndUpdateBufferAllocationLocation`` 之后**，因为
    A.10 step1 给 staging block 打的 ``c6678.dma_load`` 注解需要 schedule
    阶段的 ``cache_read + compute_at`` 已经把 staging block 嵌进对应的 outer
    loop 里——这正是 ``PlanAndUpdateBufferAllocationLocation`` 的职责。

    其它 target 不受影响（pass 内部对 ``target.kind.name == "c6678"`` 做守卫）。

    路线图见 ``Test4dsp/learning.md`` §4.8.2 / §4.8.3。

    --- A.3 ``C6678DMALegalize`` ---
    --- A.3 ``C6678DMALegalize`` ---
    紧跟 A.9 之后挂载，对 ``call_extern("load_row_major_tile", ...)`` 做纯静态
    合法性校验（scope / 正数性 / 单次传输字节上限 / 行对齐告警），只读不改写
    IR。挂在这里而不是 ``LowerOpaqueBlock`` 之后，是因为此时 call_extern 的 11
    个 args 都还是结构化的 ``IntImm / StringImm`` 形态，便于校验；后续 pass
    可能把它们折成长表达式。
    """
    return [
        tirx.transform.C6678DMALower(),
        tirx.transform.C6678DMALegalize(),
    ]


def _c6678_pre_packed_api_passes():
    """紧贴 ``MakePackedAPI`` 之前执行的 c6678 专属 pass 序列。

    顺序约束：

    1. **A.7 ``C6678LowerEntry``**：把 c6678 host PrimFunc 入口签名从
       ``int32_t __tvm_ffi_*(void*, void*, int32_t, void*)`` 降级为 bare-C 形态，
       并通过显式置 ``calling_conv = kCPackedFunc`` 让 ``MakePackedAPI`` 自动跳过。
    2. **A.8 ``C6678MulticoreLower``**：在 A.7 已经把 ``params`` 切成
       ``[A, B, C]`` 等 bare 指针之后，再追加 ``core_mask`` 形参并把
       ``ForKind.PARALLEL`` 循环降级为 ``GetCoreNum / c6678_get_core_id /
       C6678E_SyncN`` 形态的多核派发。

    其它 target 不受影响（pass 内部对 ``target.kind.name == "c6678"`` 做守卫）。

    路线图见 ``Test4dsp/learning.md`` §4.8.2 / §4.8.3。
    """
    return [
        tirx.transform.C6678LowerEntry(),
        tirx.transform.C6678MulticoreLower(),
    ]

# 把s_tir从上层代码lower到后段代码的流程固定
# 默认 pipeline + 若干配置开关
def default_s_tir_pipeline():
    """The default tirx pipeline used in tvm.tirx.build"""

    @tvm.transform.module_pass(opt_level=0)
    def _pipeline(mod: tvm.ir.IRModule, _ctx: tvm.transform.PassContext) -> tvm.ir.IRModule:
        """The default lowering passes for TIR backend."""
        pass_ctx = tvm.transform.PassContext.current()
        config = pass_ctx.config
        passes = []
        passes.extend(_c6678_specific_passes())
        passes.extend([
            s_tir.transform.CanonicalizeLoop(), # 归一化，把循环的格式统一标识
            s_tir.transform.LowerCrossThreadReduction(),    # lower跨线程归约
            s_tir.transform.LowerInitBlock(),    # 把 block 的 init 部分从高层 block 语义里拆出来
            s_tir.transform.PlanAndUpdateBufferAllocationLocation(),    # 计划并更新 buffer 分配位置
        ])
        passes.extend(_c6678_post_schedule_passes())
        passes.extend([
            s_tir.transform.ConvertBlocksToOpaque(),
            s_tir.transform.LiftThreadBinding(),
            s_tir.transform.ManifestSharedMemoryLocalStage(),
            s_tir.transform.CompactBufferAllocation(),
            s_tir.transform.LowerAutoCopy(),
            s_tir.transform.UnifyThreadBinding(),
            s_tir.transform.LowerMatchBuffer(),
            tirx.transform.Simplify(),
            # 内存布局、循环和软件流水阶段
            s_tir.transform.InjectPermutedLayout(),     # 重排布局
            s_tir.transform.AnnotateIrregularLoop(),     # 标注非规则循环
            s_tir.transform.InjectSoftwarePipeline(),     # 注入软件流水
            s_tir.transform.TransformMmaBufferLayout(),
            s_tir.transform.LowerOpaqueBlock(),
            tirx.transform.FlattenBuffer(),
            tirx.transform.BF16ComputeLegalize(),
            tirx.transform.NarrowDataType(32),
            s_tir.transform.LoopPartition(),
            tirx.transform.VectorizeLoop(not bool(config.get("tirx.disable_vectorize", False))),
            s_tir.transform.InjectVirtualThread(),
            s_tir.transform.InjectDoubleBuffer(),       # 双缓冲
        ])
        if not bool(config.get("tirx.disable_storage_rewrite", False)):
            passes.append(tirx.transform.StorageRewrite()) # 存储重写，内存复用
        # A.9 step2：StorageRewrite 会丢弃 AllocBuffer 上除 kVolatile 之外的注解，
        # 因此 C6678AnnotateL2Alloc 必须放在 StorageRewrite 之后、LowerTVMBuiltin
        # 之前；它给 scope='global.l2' 的 AllocBuffer 追加 disable_lower_builtin=True，
        # 使 LowerTVMBuiltin 直接保留这条 alloc 而不去查 device-id。
        passes.append(tirx.transform.C6678AnnotateL2Alloc())
        # PR-S2 phase A：再给所有 scope='global' 的 AllocBuffer 加同一注解，覆盖
        # softmax / layernorm 等带中间 alloc 的 PrimFunc（matmul 没中间 alloc 不受影响）。
        # 必须放在 C6678AnnotateL2Alloc 之后（那一步已经把 global.l2 改写成 global），
        # 否则 global.l2 的 alloc 会因没追加注解而走回 workspace 路径。
        passes.append(tirx.transform.C6678AnnotateGlobalAlloc())
        if config.get("tirx.use_async_copy", False):
            passes.append(s_tir.transform.LowerAsyncDMA())
        passes.extend(
            [
                s_tir.transform.HoistIfThenElse(),      # 条件语句尽量外提
                tirx.transform.UnrollLoop(),
                s_tir.transform.RenormalizeSplitPattern(),
                tirx.transform.Simplify(),
                tirx.transform.RemoveNoOp(),            # 移除无用语句
                s_tir.transform.RewriteUnsafeSelect(),  # 重写不安全语句：非法访存等
            ]
        )
        # Additional passes based on configuration.
        if bool(config.get("tirx.instrument_bound_checkers", False)):
            passes.append(s_tir.transform.InstrumentBoundCheckers())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32(True))
        if not bool(config.get("tirx.disable_cse_tir", False)):
            passes.append(tirx.transform.CommonSubexprElim())
        if bool(config.get("tirx.instrument_lwp", False)):
            passes.append(s_tir.transform.InstrumentProfileIntrinsics())
        passes.extend(
            [
                # Bind the target first so that target-specific attributes are available.
                tirx.transform.FP8ComputeLegalize(),
                # VerifyVTCMLimit must occur before LowerVtcmAlloc.
                s_tir.transform.VerifyVTCMLimit(),
                s_tir.transform.LowerVtcmAlloc(),
                tirx.transform.VerifyMemory(),
                tirx.transform.AnnotateEntryFunc(),
            ]
        )
        # 针对线程并行和共享内存协作的 lowering
        passes.extend(
            [
                s_tir.transform.ThreadSync("shared"),
                s_tir.transform.ThreadSync("shared.dyn"),
                s_tir.transform.ThreadSync("warp"),
                s_tir.transform.InferFragment(),
                s_tir.transform.LowerThreadAllreduce(),
            ]
        )
        if bool(config.get("tirx.use_async_copy", False)):
            passes.append(s_tir.transform.InjectPTXAsyncCopy())
        if bool(config.get("tirx.ptx_ldg32", False)):
            passes.append(s_tir.transform.InjectPTXLDG32())
        passes.extend(
            [
                tirx.transform.AnnotateDeviceRegions(),
                tirx.transform.SplitHostDevice(),
                # MergeSharedMemoryAllocations must follow SplitHostDevice.
                s_tir.transform.MergeSharedMemoryAllocations(),
            ]
        )
        # A.7：紧贴 MakePackedAPI 之前对 c6678 host func 做 bare-C entry 改写，
        # 通过显式置 calling_conv=kCPackedFunc 让随后的 MakePackedAPI 跳过本函数。
        passes.extend(_c6678_pre_packed_api_passes())
        # ABI 生成与最终 lowering
        passes.extend(
            [
                tirx.transform.MakePackedAPI(),
                tirx.transform.FP8StorageLegalize(),
                tirx.transform.BF16StorageLegalize(),
                tirx.transform.LowerDeviceKernelLaunch(),
            ]
        )
        mod = tvm.ir.transform.Sequential(passes)(mod)
        return mod

    return _pipeline


def finalize_host_passes():  # pylint: disable=unused-argument
    """The default finalization passes for TIR backend."""
    host_pass_list = [
        tirx.transform.LowerTVMBuiltin(),
        tirx.transform.LowerCustomDatatypes(),
        tirx.transform.LowerIntrin(),
    ]
    return tvm.ir.transform.Sequential(host_pass_list)


tir_pipeline.PIPELINE_MAP["s_tir"] = default_s_tir_pipeline
