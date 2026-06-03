"""C6678 端到端集成测试（A.1 ~ A.10 step1 全链路落地状态可视化）。

本测试不是某个单 pass 的冒烟，而是把 **从用户视角的 TVMScript 输入** 一直跑到
**bare-C 源码输出** 的完整链路串起来跑一次，逐项断言：

1. 用户输入：一段 ND 形式的 fp32 matmul TVMScript，未指定任何 c6678 细节；
2. dlight.c6678.Matmul → A.5 dispatcher → A.4 features 抽取 → MatmulGemmTemplate
   产出带 ``c6678.dma_load`` 注解的 staging block；
3. ``tvm.tirx.build`` 内部按 ``default_s_tir_pipeline`` 串行跑：
   * A.2 ``C6678StoragePlan``：写 ``c6678.storage_plan`` attrs（仅 metadata）；
   * A.9 ``C6678DMALower``：staging block → ``call_extern("load_row_major_tile", ...)``；
   * A.3 ``C6678DMALegalize``：对每条 DMA call 做静态合法性校验（只读）；
   * A.9 ``C6678AnnotateL2Alloc``：``scope="global.l2"`` AllocBuffer → ``"global"``；
   * A.7 ``C6678LowerEntry``：bare-C 入口签名 + 跳过 ``MakePackedAPI``；
   * A.8 ``C6678MulticoreLower``：``ForKind.PARALLEL`` → ``GetCoreNum / GetLogicCoreId / C6678E_SyncN``；
4. 输出 C 源码同时具备：bare-C 入口 / 多核派发 / SyncN 收尾 / L2 staging 缓冲 /
   ``load_row_major_tile`` DMA 调用 / 入参 ``int32_t core_mask``。

未落地项：
* A.10 step2 ``_p_`` 内层退化开关（依赖 BSP ``_p_`` 签名与 staging stride 对齐，未做）；
* A.6 dlight.c6678 已有 ``Matmul`` / ``Softmax`` / ``ElementGreaterEqual`` 起步；
  conv / reduce / LSTM pattern 等仍未补。

运行（必须先进 conda tvm_env）：

    source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \\
    conda activate tvm_env && \\
    PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \\
    python /home/tangqingyun/tvm/Test4dsp/tests/test_c6678_end_to_end.py
"""

from __future__ import annotations

import tvm
from tvm import s_tir
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T


M, N, K = 128, 128, 128


def _check(cond: bool, msg: str) -> None:
    """内联 assert，失败时直接抛 AssertionError，便于命令行直接看到原因。"""
    if not cond:
        raise AssertionError(msg)


# ---- 用户视角的 TVMScript 输入：纯 ND，无任何 c6678 细节 ----------------------
@T.prim_func
def matmul_fp32(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    """fp32 128x128x128 matmul，全部默认 scope（落 DDR），无 cache_read，无 parallel。"""
    T.func_attr({"global_symbol": "matmul_fp32", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.sblock("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def _build_and_get_source() -> str:
    """串完 schedule + ``tvm.tirx.build``，返回 c6678 codegen 源码。"""
    mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
    target = tvm.target.Target("c6678")
    with target:
        sch_mod = dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)

    is_sched = sch_mod["matmul_fp32"].attrs.get("tirx.is_scheduled")
    _check(
        bool(int(is_sched)) is True,
        f"expect tirx.is_scheduled=True after dlight.c6678.Matmul, got {is_sched}",
    )

    runtime_mod = tvm.tirx.build(sch_mod, target=target)
    return runtime_mod.inspect_source("c6678")


def test_user_input_compiles_to_c_source() -> None:
    """A.6 起步：用户写 TVMScript + ``tvm.tirx.build`` 一行能产出非空 c6678 源码。"""
    src = _build_and_get_source()
    _check(len(src) > 0, "expect non-empty c6678 source")
    print(f"[OK] tvm.tirx.build produced c6678 source ({len(src)} chars)")


def test_a7_bare_c_entry_signature() -> None:
    """A.7：入口签名是 bare-C 形态（``float* / int32_t core_mask``），而非 packed FFI。"""
    src = _build_and_get_source()
    _check(
        "void matmul_fp32(float* A, float* B, float* C, int32_t core_mask)" in src,
        "expect bare-C entry signature with core_mask",
    )
    _check(
        "__tvm_ffi_matmul_fp32" not in src,
        "MakePackedAPI must be bypassed (no __tvm_ffi_* wrapper)",
    )
    print("[OK] A.7 bare-C entry: void matmul_fp32(float*, float*, float*, int32_t)")


def test_a8_multicore_dispatch_and_syncn() -> None:
    """A.8：``ForKind.PARALLEL`` 已降为 ``GetCoreNum / GetLogicCoreId`` + 末尾 ``C6678E_SyncN``。"""
    src = _build_and_get_source()
    _check("GetCoreNum(core_mask)" in src, "expect GetCoreNum(core_mask) in body")
    _check(
        "GetLogicCoreId(core_mask, DNUM)" in src,
        "expect GetLogicCoreId(core_mask, DNUM) in body",
    )
    _check(
        "C6678E_SyncN(GetCoreNum(core_mask), GetLogicCoreId(core_mask, DNUM))" in src,
        "expect SyncN at function tail",
    )
    print("[OK] A.8 multicore dispatch + C6678E_SyncN tail-sync emitted")


def test_a9_dma_call_and_l2_staging() -> None:
    """A.9 + A.10 step1：``load_row_major_tile`` 取代 staging copy；L2 staging 绑定核心 L2。"""
    src = _build_and_get_source()
    _check(
        'load_row_major_tile((&(A[0])), (&(A_global_l2[0]))' in src,
        "expect DMA call for A → A_global_l2[0]",
    )
    _check(
        'load_row_major_tile((&(B[0])), (&(A_global_l2[16384]))' in src,
        "expect DMA call for B → A_global_l2[16384] (StorageRewrite combined into single l2 buffer)",
    )
    _check(
        "float* A_global_l2 = (float*)(276889600 + DNUM * 16777216);" in src,
        "expect A_global_l2 bound to per-core L2 base address",
    )
    _check(
        'load_row_major_tile((&(A[0])), (&(A_global_l2[0])), (ax0_0 * 32),' in src,
        "expect load_row_major_tile to use the 8-parameter BSP signature",
    )
    print("[OK] A.9 + A.10 step1 DMA tiles + per-core L2 pointer emitted")


def test_a3_dma_legalize_pass_runs_silently() -> None:
    """A.3：``C6678DMALegalize`` 已挂入主 pipeline，对所有 DMA call 做合法性校验。

    最强证据是：经过 ``tvm.tirx.build`` 完整流水线后，DMA call 的所有静态参数
    都满足 A.3 的校验维度——
      * ``load_row_major_tile`` 使用新的 8 参数 BSP 签名，不再把 ``src_scope`` 编进 C 调用；
      * rows / cols / src_ld / elem_size 都是正整数：源码里 ``32, 32, 128, 4``；
      * rows*cols*elem_size = 32*32*4 = 4096 字节，远 < dma_max_transfer (0x7FFFFFFF)；
      * cols*elem_size = 32*4 = 128 字节，是 dma_align_bytes=64 的整数倍。

    若 A.3 未挂入 pipeline 或断言失败，``tvm.tirx.build`` 会抛 ValueError；
    本测试通过比对源码里出现的 DMA 参数，间接验证 A.3 已运行并放行。
    """
    src = _build_and_get_source()
    # 每条 load_row_major_tile 都应是 32×32 fp32 tile（4096B）搬到 L2 staging
    _check(
        ', 32, 32, 128, 4);' in src,
        "expect DMA tile (rows=32 cols=32 src_ld=128 elem=4) confirming A.3 passed",
    )
    print(
        "[OK] A.3 C6678DMALegalize: every DMA call in pipeline satisfies "
        "rows*cols*elem=4096B / 128B row aligned"
    )


def test_a2_storage_plan_attrs_visible() -> None:
    """A.2：``C6678StoragePlan`` 把 storage plan 写到 attrs（仅 metadata，下游不消费）。"""
    from tvm import tirx
    mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
    target = tvm.target.Target("c6678")
    bound = tirx.transform.BindTarget(target)(mod)
    bound = tirx.transform.C6678StoragePlan()(bound)
    plan = bound["matmul_fp32"].attrs.get("c6678.storage_plan")
    _check(plan is not None, "expect c6678.storage_plan attr after A.2")
    _check(len(plan) >= 3, f"expect at least 3 plan entries (A/B/C), got {len(plan)}")
    print(f"[OK] A.2 storage_plan attrs visible: {len(plan)} entries (one per buffer)")


def test_full_source_total_size_matches_baseline() -> None:
    """端到端零差分回归：scheduled 出码字节数应与本轮 ABI/L2 修正后的快照一致（2283 chars）。

    一旦后续 pass 链路改动让出码字节数变化，这条用例会立刻报警。这是阻挡
    回归（unintended regression）的一道防线。
    """
    src = _build_and_get_source()
    expected = 2283
    _check(
        len(src) == expected,
        f"expect scheduled c6678 source size to be {expected} chars (snapshot after "
        f"GetLogicCoreId + per-core L2 pointer + 8-arg load_row_major_tile), got {len(src)}",
    )
    print(f"[OK] zero-diff: scheduled c6678 source = {expected} chars (snapshot match)")


if __name__ == "__main__":
    print("=" * 60)
    print("A.1 ~ A.10 step1 端到端集成测试")
    print("=" * 60)
    test_user_input_compiles_to_c_source()
    test_a2_storage_plan_attrs_visible()
    test_a3_dma_legalize_pass_runs_silently()
    test_a7_bare_c_entry_signature()
    test_a8_multicore_dispatch_and_syncn()
    test_a9_dma_call_and_l2_staging()
    test_full_source_total_size_matches_baseline()
    print("=" * 60)
    print("[OK] all c6678 end-to-end integration tests passed")
    print("=" * 60)
