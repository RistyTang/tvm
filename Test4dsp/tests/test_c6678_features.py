"""A.4 ``tvm.tirx.analysis.extract_features`` 冒烟测试。

不依赖 pytest 框架（pytest plugin 需要 LLVM 支持，本地 build 没有），可以直接
``python Test4dsp/tests/test_c6678_features.py`` 跑。

覆盖三类形态：

1. fp32 128x128x128 matmul（端到端 demo 同形态）—— 验证 ``op_kind=="gemm"`` +
   ``tile_hint==(32,32,32)`` + ``static_alloc_l2_bytes==8192``。
2. 非 c6678 target —— ``extract_features`` 返回 ``None``。
3. 单输入逐点 ``y=x*2`` —— ``op_kind=="elementwise"``，无 reduce。
"""

from __future__ import annotations

import os
import sys

import tvm
from tvm.script import tirx as T
from tvm.tirx.analysis import (
    C6678OpFeatures,
    C6678PrimFuncFeatures,
    extract_features,
)


@T.prim_func
def _matmul_fp32(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"global_symbol": "matmul_fp32", "tir.noalias": True})
    for i, j, k in T.grid(128, 128, 128):
        with T.sblock("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


@T.prim_func
def _scale_fp32(
    X: T.Buffer((128, 128), "float32"),
    Y: T.Buffer((128, 128), "float32"),
):
    T.func_attr({"global_symbol": "scale_fp32", "tir.noalias": True})
    for i, j in T.grid(128, 128):
        with T.sblock("scale"):
            vi, vj = T.axis.remap("SS", [i, j])
            Y[vi, vj] = X[vi, vj] * T.float32(2.0)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        print(f"[FAIL] {msg}")
        sys.exit(1)


def test_matmul_fp32():
    target = tvm.target.Target("c6678")
    feats = extract_features(_matmul_fp32, target, func_name="matmul_fp32")
    _check(isinstance(feats, C6678PrimFuncFeatures), "expect C6678PrimFuncFeatures")
    _check(len(feats.blocks) == 1, f"expect 1 block, got {len(feats.blocks)}")
    op = feats.blocks[0]
    _check(isinstance(op, C6678OpFeatures), "expect C6678OpFeatures")
    _check(op.op_kind == "gemm", f"expect op_kind=='gemm', got {op.op_kind}")
    _check(op.dom_kind == "SSR", f"expect dom_kind=='SSR', got {op.dom_kind}")
    _check(op.dom_extents == (128, 128, 128), f"got dom_extents={op.dom_extents}")
    _check(op.dtype == "float32", f"got dtype={op.dtype}")
    _check(op.is_static_shape, "expect is_static_shape")
    _check(op.flop_count_static == 128 * 128 * 128 * 2, f"got flops={op.flop_count_static}")
    _check(op.tile_hint == (32, 32, 32), f"got tile_hint={op.tile_hint}")
    # 当前 matmul schedule 的 cache_read 最终会 staging A/B 两个 128x128 fp32 buffer。
    _check(op.static_alloc_l2_bytes == 131072, f"got alloc={op.static_alloc_l2_bytes}")
    _check(feats.config is not None and feats.config.l2_size == 1024 * 1024,
           f"got l2_size={None if feats.config is None else feats.config.l2_size}")
    print(
        f"[OK] matmul_fp32: op_kind={op.op_kind} dom={op.dom_extents} "
        f"tile_hint={op.tile_hint} alloc={op.static_alloc_l2_bytes}B "
        f"flops={op.flop_count_static}"
    )


def test_non_c6678_returns_none():
    target = tvm.target.Target("llvm")
    feats = extract_features(_matmul_fp32, target)
    _check(feats is None, f"expect None on non-c6678 target, got {feats}")
    print("[OK] non-c6678 target returns None")


def test_elementwise():
    target = tvm.target.Target("c6678")
    feats = extract_features(_scale_fp32, target, func_name="scale_fp32")
    _check(feats is not None, "expect features for elementwise")
    op = feats.blocks[0]
    _check(op.op_kind == "elementwise", f"expect op_kind=='elementwise', got {op.op_kind}")
    _check(op.dom_kind == "SS", f"expect dom_kind=='SS', got {op.dom_kind}")
    _check(op.flop_count_static == 128 * 128, f"got flops={op.flop_count_static}")
    # elementwise 不命中 gemm tile_hint，应为全 None
    _check(all(t is None for t in op.tile_hint), f"got tile_hint={op.tile_hint}")
    _check(op.static_alloc_l2_bytes is None, f"got alloc={op.static_alloc_l2_bytes}")
    print(f"[OK] scale_fp32: op_kind={op.op_kind} dom={op.dom_extents}")


if __name__ == "__main__":
    test_matmul_fp32()
    test_non_c6678_returns_none()
    test_elementwise()
    print("[OK] all c6678 features tests passed")
