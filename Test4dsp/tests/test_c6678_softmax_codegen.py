"""C6678 softmax 端到端集成测试（PR-S2 phase A）。

本测试串联 PrimFunc 输入 → ``dlight.c6678.Softmax`` schedule → ``tvm.tirx.build``
→ c6678 bare-C 源码，逐项断言关键产物：

1. PrimFunc 输入：4 块 fp32 softmax，``["SR", "SS", "SR", "SS"]`` + 静态 shape，
   用 ``T.sblock_alloc_buffer`` 分配 ``T_max / T_exp / T_sum``，``T.exp(...)``
   走 ``tirx.exp`` 默认 lowering（fp32 → ``expf``）；
2. ``dlight.c6678.Softmax`` 应用：``parallel(outer) + 逆序 compute_at(前 3 块)``
   把 4 块塞进同一 row loop，并设上 ``tirx.is_scheduled``；
3. ``tvm.tirx.build`` 完整流水线（含 PR-S2 phase A 新增的
   ``C6678AnnotateGlobalAlloc``）：让中间 ``T_max / T_exp / T_sum``
   ``AllocBuffer`` 直接以栈数组形式落地，避开 ``LowerTVMBuiltin`` workspace 路径；
4. 最终 c6678 源码具备：bare-C 入口签名 / 多核派发 / SyncN 收尾 /
   4 个 ax1 内层循环串行（max → exp → sum → div）/ ``expf(...)`` 调用 /
   3 个中间栈数组（``T_max[1] / T_exp[1024] / T_sum[1]``）。

运行（必须先进 conda tvm_env）：

    source /home/tangqingyun/miniconda3/etc/profile.d/conda.sh && \\
    conda activate tvm_env && \\
    PYTHONPATH=/home/tangqingyun/tvm/python:/home/tangqingyun/tvm/3rdparty/tvm-ffi/python \\
    python /home/tangqingyun/tvm/Test4dsp/tests/test_c6678_softmax_codegen.py
"""

from __future__ import annotations

import tvm
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T


OUTER = 8
INNER = 1024


def _check(cond: bool, msg: str) -> None:
    """内联 assert，失败时直接抛 AssertionError，便于命令行直接看到原因。"""
    if not cond:
        raise AssertionError(msg)


# ---- 用户视角的 TVMScript 输入：4 块数值稳定 fp32 softmax -------------------
@T.prim_func
def softmax_fp32(
    A: T.Buffer((OUTER, INNER), "float32"),
    Out: T.Buffer((OUTER, INNER), "float32"),
):
    """fp32 softmax (axis=-1)：Out[i, k] = exp(A[i, k] - max_k) / sum_k exp(...)"""
    T.func_attr({"global_symbol": "softmax_fp32", "tir.noalias": True})
    # 必须用 sblock_alloc_buffer：进 root SBlock 的 alloc_buffers，避免
    # compute_at 把它们当成 output（详见 dlight/c6678/softmax.py 模块 docstring）。
    T_max = T.sblock_alloc_buffer((OUTER,), "float32")
    T_exp = T.sblock_alloc_buffer((OUTER, INNER), "float32")
    T_sum = T.sblock_alloc_buffer((OUTER,), "float32")

    for i, k in T.grid(OUTER, INNER):
        with T.sblock("max"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                T_max[vi] = T.float32(-3.4028234e38)
            T_max[vi] = T.max(T_max[vi], A[vi, vk])

    for i, k in T.grid(OUTER, INNER):
        with T.sblock("exp"):
            vi, vk = T.axis.remap("SS", [i, k])
            T_exp[vi, vk] = T.exp(A[vi, vk] - T_max[vi])

    for i, k in T.grid(OUTER, INNER):
        with T.sblock("sum"):
            vi, vk = T.axis.remap("SR", [i, k])
            with T.init():
                T_sum[vi] = T.float32(0)
            T_sum[vi] = T_sum[vi] + T_exp[vi, vk]

    for i, k in T.grid(OUTER, INNER):
        with T.sblock("div"):
            vi, vk = T.axis.remap("SS", [i, k])
            Out[vi, vk] = T_exp[vi, vk] / T_sum[vi]


def _build_and_get_source() -> str:
    """串完 schedule + ``tvm.tirx.build``，返回 c6678 codegen 源码。"""
    mod = tvm.IRModule({"softmax_fp32": softmax_fp32})
    target = tvm.target.Target("c6678")
    with target:
        sch_mod = dlight.ApplyDefaultSchedule(c6678_dlight.Softmax())(mod)

    is_sched = sch_mod["softmax_fp32"].attrs.get("tirx.is_scheduled")
    _check(
        bool(int(is_sched)) is True,
        f"expect tirx.is_scheduled=True after dlight.c6678.Softmax, got {is_sched}",
    )

    runtime_mod = tvm.tirx.build(sch_mod, target=target)
    return runtime_mod.inspect_source("c6678")


def test_user_input_compiles_to_c_source() -> None:
    """端到端起步：用户写 TVMScript + ``tvm.tirx.build`` 一行能产出非空 c6678 源码。"""
    src = _build_and_get_source()
    _check(len(src) > 0, "expect non-empty c6678 source")
    print(f"[OK] tvm.tirx.build produced c6678 source ({len(src)} chars)")


def test_a7_bare_c_entry_signature() -> None:
    """A.7：入口签名是 bare-C 形态（``float* / int32_t core_mask``），而非 packed FFI。"""
    src = _build_and_get_source()
    _check(
        "void softmax_fp32(float* A, float* Out, int32_t core_mask)" in src,
        "expect bare-C entry signature with core_mask",
    )
    _check(
        "__tvm_ffi_softmax_fp32" not in src,
        "MakePackedAPI must be bypassed (no __tvm_ffi_* wrapper)",
    )
    print("[OK] A.7 bare-C entry: void softmax_fp32(float*, float*, int32_t)")


def test_a8_multicore_dispatch_and_syncn() -> None:
    """A.8：``parallel(ax0)`` 已降为 ``GetCoreNum / c6678_get_core_id`` + 末尾 ``C6678E_SyncN``。"""
    src = _build_and_get_source()
    _check("GetCoreNum(core_mask)" in src, "expect GetCoreNum(core_mask) in body")
    _check(
        "c6678_get_core_id(core_mask)" in src,
        "expect c6678_get_core_id(core_mask) in body",
    )
    _check(
        "C6678E_SyncN(GetCoreNum(core_mask), c6678_get_core_id(core_mask))" in src,
        "expect SyncN at function tail",
    )
    print("[OK] A.8 multicore dispatch + C6678E_SyncN tail-sync emitted")


def test_intermediate_buffers_emitted_as_stack_arrays() -> None:
    """PR-S2 phase A 新增 ``C6678AnnotateGlobalAlloc`` 兜底：3 个中间 buffer 以栈数组形式落地。

    经过 ``compute_at(...) + parallel(ax0)`` 后，中间 buffer 的最外层 row 维度
    被吃掉，每个核每 row 只需 ``T_max[1] / T_exp[1024] / T_sum[1]``。如果
    ``C6678AnnotateGlobalAlloc`` 没生效，``LowerTVMBuiltin`` 会因 32KB 总量
    超过 ``kMaxStackAlloca=1024`` 把 alloc 改写成 ``TVMBackendAllocWorkspace``
    并要求 device-id（c6678 是 bare-metal kDLCPU 没这个信息），整个 build 会
    抛 ``Unknown device id in current IR``。
    """
    src = _build_and_get_source()
    _check("float T_max[1];" in src, "expect T_max stack array (compute_at 收掉 outer 后 size=1)")
    _check("float T_exp[1024];" in src, "expect T_exp stack array (size=INNER=1024)")
    _check("float T_sum[1];" in src, "expect T_sum stack array (compute_at 收掉 outer 后 size=1)")
    _check(
        "TVMBackendAllocWorkspace" not in src,
        "expect no workspace runtime call (c6678 bare-metal has no workspace)",
    )
    print("[OK] PR-S2 phase A: T_max[1] / T_exp[1024] / T_sum[1] stack arrays emitted")


def test_softmax_3pass_serial_structure() -> None:
    """Softmax rule：每 row 内串行 4 个 ax1 循环（max → exp → sum → div）。

    schedule 把前 3 块逆序 ``compute_at`` 到 epilogue 的 ``ax0`` 下，所以源码里
    应该出现 4 个独立的 ``for (ax1 ...; ax1 < 1024; ++ax1)`` 内循环（其中 3 个
    被重命名为 ax1_1 / ax1_2 / ax1_3）。
    """
    src = _build_and_get_source()
    for tag in ["ax1 = 0;", "ax1_1 = 0;", "ax1_2 = 0;", "ax1_3 = 0;"]:
        _check(tag in src, f"expect inner loop counter {tag} (3-pass softmax serial structure)")
    print("[OK] 4-pass inner loop structure: max → exp → sum → div all present")


def test_expf_lowering_via_default_intrin() -> None:
    """``T.exp`` 走 ``tirx.exp`` 默认 lowering：fp32 → C99 ``expf(...)``。

    本断言同时验证：c6678 codegen 不需要任何 c6678 专属 intrin 注册，靠
    ``register_intrin_lowering("tirx.exp", target="default", _rule_float_suffix)``
    就能拿到 ``expf``。如果 lowering 没生效，会出现 ``__tvm_exp`` 之类符号。
    """
    src = _build_and_get_source()
    _check("expf(" in src, "expect C99 expf(...) call from tirx.exp default lowering")
    _check(
        "__tvm_exp" not in src and "tirx.exp" not in src,
        "expect tirx.exp fully lowered (no symbolic op left in source)",
    )
    print("[OK] tirx.exp → expf(...) (C99) via target='default' intrin lowering")


def test_full_source_total_size_matches_baseline() -> None:
    """端到端零差分回归：scheduled 出码字节数应与 PR-S2 phase A 落地快照一致（1925 chars）。

    一旦后续 pass 链路改动让出码字节数变化，这条用例会立刻报警，作为
    阻挡 unintended regression 的防线（参考 matmul end_to_end 的 2345 chars 防线）。
    """
    src = _build_and_get_source()
    expected = 1925
    _check(
        len(src) == expected,
        f"expect scheduled c6678 softmax source size to be {expected} chars (snapshot "
        f"PR-S2 phase A landed), got {len(src)}",
    )
    print(f"[OK] zero-diff: scheduled c6678 softmax source = {expected} chars (snapshot match)")


if __name__ == "__main__":
    print("=" * 60)
    print("PR-S2 phase A: c6678 softmax 端到端集成测试")
    print("=" * 60)
    test_user_input_compiles_to_c_source()
    test_a7_bare_c_entry_signature()
    test_a8_multicore_dispatch_and_syncn()
    test_intermediate_buffers_emitted_as_stack_arrays()
    test_softmax_3pass_serial_structure()
    test_expf_lowering_via_default_intrin()
    test_full_source_total_size_matches_baseline()
    print("=" * 60)
    print("[OK] all c6678 softmax codegen tests passed")
    print("=" * 60)
