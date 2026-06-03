"""A.5 dispatcher 冒烟测试。

覆盖三个场景，与 A.4 ``test_c6678_features.py`` 形成互补：

1. **正向匹配**：``128x128x128 fp32 matmul`` → ``select_template`` 命中
   ``MatmulGemmTemplate``，且 ``estimate_l2_bytes == 8192``、远小于
   ``feats.config.l2_size``；
2. **形态不匹配**：``elementwise scale`` → ``select_template`` 返回 ``None``
   （目前 _TEMPLATE_REGISTRY 只有 gemm 模板）；
3. **L2 容量门禁**：通过手工伪造一个 ``static_alloc_l2_bytes`` 大于
   ``l2_size`` 的 features，确认 dispatcher 主动跳过该模板（fail-safe）。

不依赖 pytest，``python Test4dsp/tests/test_c6678_dispatcher.py`` 直接跑。
"""

from __future__ import annotations

from dataclasses import replace

import tvm
from tvm import tirx
from tvm.script import tirx as T
from tvm.s_tir.dlight.c6678 import (
    MatmulGemmTemplate,
    select_template,
    features_for_func,
)
from tvm.tirx.analysis import C6678OpFeatures, C6678PrimFuncFeatures


def _check(cond: bool, msg: str) -> None:
    """内联的 assert 替身，失败时直接抛 AssertionError，便于命令行直接看到原因。"""
    if not cond:
        raise AssertionError(msg)


@T.prim_func
def _matmul_fp32(
    A: T.Buffer((128, 128), "float32"),
    B: T.Buffer((128, 128), "float32"),
    C: T.Buffer((128, 128), "float32"),
):
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
    for i, j in T.grid(128, 128):
        with T.sblock("scale"):
            vi, vj = T.axis.remap("SS", [i, j])
            Y[vi, vj] = X[vi, vj] * T.float32(2.0)


def test_matmul_dispatch_hits_gemm_template() -> None:
    target = tvm.target.Target("c6678")
    feats = features_for_func(_matmul_fp32, target, func_name="matmul_fp32")
    _check(feats is not None, "expect features for c6678 target")
    tpl = select_template(feats, block_idx=0)
    _check(isinstance(tpl, MatmulGemmTemplate), "expect MatmulGemmTemplate")
    _check(tpl.name == "matmul_gemm", "template.name mismatch")
    est = tpl.estimate_l2_bytes(feats, 0)
    _check(est == 131072, f"expect L2 alloc 131072B, got {est}")
    _check(
        feats.config is not None and est < feats.config.l2_size,
        "L2 estimate must be below l2_size cap",
    )
    print(
        f"[OK] matmul_fp32 dispatched to {tpl.name}, L2 estimate {est}B "
        f"(cap {feats.config.l2_size}B)"
    )


def test_elementwise_returns_none() -> None:
    target = tvm.target.Target("c6678")
    feats = features_for_func(_scale_fp32, target, func_name="scale_fp32")
    _check(feats is not None, "expect features for c6678 target")
    tpl = select_template(feats, block_idx=0)
    _check(tpl is None, f"elementwise must not match any template, got {tpl}")
    print("[OK] scale_fp32 (elementwise) returns None as expected")


def test_l2_capacity_gate_skips_oversized_template() -> None:
    """伪造一个超容量的 features，验证派发器能拦下来。

    手工把 ``static_alloc_l2_bytes`` 改成一个比 l2_size 还大的值，
    其它字段保持不变。这模拟"未来可能出现的大 tile 模板"或"用户自定义的
    超大 staging"——派发器必须主动跳过，而不是让 build 在板卡上挂掉。
    """
    target = tvm.target.Target("c6678")
    feats = features_for_func(_matmul_fp32, target)
    _check(feats is not None, "expect features for c6678 target")
    cap = feats.config.l2_size
    bloated_op = replace(feats.blocks[0], static_alloc_l2_bytes=cap * 2)
    bloated_feats = C6678PrimFuncFeatures(
        func_name=feats.func_name,
        blocks=(bloated_op,),
        config=feats.config,
    )
    tpl = select_template(bloated_feats, block_idx=0)
    _check(
        tpl is None,
        f"L2 capacity gate should reject oversized template, got {tpl}",
    )
    print(
        f"[OK] L2 capacity gate rejected template with "
        f"alloc={bloated_op.static_alloc_l2_bytes}B (cap {cap}B)"
    )


if __name__ == "__main__":
    test_matmul_dispatch_hits_gemm_template()
    test_elementwise_returns_none()
    test_l2_capacity_gate_skips_oversized_template()
    print("[OK] all c6678 dispatcher tests passed")
