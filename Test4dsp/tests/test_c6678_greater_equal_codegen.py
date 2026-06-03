"""C6678 ElementGreaterEqual codegen smoke test."""

from __future__ import annotations

import tvm
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T


N = 262144


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


@T.prim_func
def greater_equal_fp32(
    A: T.Buffer((N,), "float32"),
    B: T.Buffer((N,), "float32"),
    Out: T.Buffer((N,), "bool"),
):
    """Same-shape greater_equal: Out[i] = A[i] >= B[i]."""
    T.func_attr({"global_symbol": "greater_equal_fp32", "tir.noalias": True})
    for i in T.serial(N):
        with T.sblock("greater_equal"):
            vi = T.axis.spatial(N, i)
            Out[vi] = A[vi] >= B[vi]


def _build_and_get_source() -> str:
    mod = tvm.IRModule({"greater_equal_fp32": greater_equal_fp32})
    target = tvm.target.Target("c6678")
    with target:
        sch_mod = dlight.ApplyDefaultSchedule(c6678_dlight.ElementGreaterEqual())(mod)

    is_sched = sch_mod["greater_equal_fp32"].attrs.get("tirx.is_scheduled")
    _check(
        bool(int(is_sched)) is True,
        f"expect tirx.is_scheduled=True after ElementGreaterEqual, got {is_sched}",
    )

    runtime_mod = tvm.tirx.build(sch_mod, target=target)
    return runtime_mod.inspect_source("c6678")


def test_greater_equal_compiles_to_c_source() -> None:
    src = _build_and_get_source()
    _check(len(src) > 0, "expect non-empty c6678 source")
    _check(
        "void greater_equal_fp32(float* A, float* B, int8_t* Out, int32_t core_mask)" in src,
        "expect bare-C greater_equal entry with int8_t bool storage and core_mask",
    )
    _check(
        "<= A_global_l2[" in src and "Out[" in src,
        "expect generated comparison equivalent to A[i] >= B[i]",
    )
    _check("dma_trans" in src, "expect dma_trans calls for input L2 staging")
    _check(
        "dma_trans((&(A[" in src and "dma_trans((&(B[" in src,
        "expect dma_trans source pointers to use tile-dependent input offsets",
    )
    _check(
        "float A_global_l2[174080]" in src,
        "expect compact merged L2 staging storage for two 87040-element tiles",
    )
    _check(
        "float A_global_l2[524288]" not in src,
        "expect no full-size L2 allocation after dma_trans compact",
    )
    _check(
        "dma_trans((&(A[" in src and "(&(A_global_l2[0]))" in src,
        "expect first dma_trans destination to start at compact L2 offset 0",
    )
    _check(
        "dma_trans((&(B[" in src and "(&(A_global_l2[87040]))" in src,
        "expect second dma_trans destination to start at compact L2 offset 87040",
    )
    _check(
        "A_global_l2[ax0_1]" in src,
        "expect compute to use tile-local L2 indices",
    )
    _check("GetCoreNum(core_mask)" in src, "expect multicore dispatch")
    _check("C6678E_SyncN" in src, "expect tail sync")
    _check("__tvm_ffi_greater_equal_fp32" not in src, "expect no packed FFI wrapper")
    print(f"[OK] ElementGreaterEqual c6678 source generated ({len(src)} chars)")


if __name__ == "__main__":
    test_greater_equal_compiles_to_c_source()
