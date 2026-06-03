"""C6678 端到端 softmax demo（PR-S2 phase A）。

用户视角：

1. 用 ``@T.prim_func`` 写一个数值稳定的 4 块 softmax（fp32, [O, I] 形态，
   axis=-1）；
2. ``tvm.target.Target("c6678")`` 拿到目标；
3. ``dlight.ApplyDefaultSchedule(c6678.Softmax())`` 套上 PR-S2 phase A 的
   极简 schedule（``parallel(outer) + compute_at``）；
4. ``tvm.tirx.build`` 完成 IR→TIR→codegen，吐出 c6678 C 源码（含
   ``expf(...)`` 调用）；
5. baseline 与 scheduled 各 dump 一份，作为后续 phase B 加 dma_trans staging
   时的对比基线。

后续 PR-S2 phase B 落 dma_trans staging 时，本 demo 不应改接口形态，只
会让生成的 C 源码出现 ``dma_trans(...)`` 调用与 L2 staging buffer。
"""

from __future__ import annotations

import os
import sys
import traceback

import tvm
from tvm import s_tir
from tvm.s_tir import dlight
from tvm.s_tir.dlight import c6678 as c6678_dlight
from tvm.script import tirx as T

OUTER = 8
INNER = 1024

_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
SCHEDULED_OUT = os.path.join(_OUT_DIR, "generated_c6678_softmax_via_build.c")
BASELINE_OUT = os.path.join(_OUT_DIR, "generated_c6678_softmax_baseline.c")


@T.prim_func
def softmax_fp32(
    A: T.Buffer((OUTER, INNER), "float32"),
    Out: T.Buffer((OUTER, INNER), "float32"),
):
    """fp32 数值稳定 softmax：Out[i, k] = exp(A[i, k] - max_k) / sum_k exp(...)"""
    T.func_attr({"global_symbol": "softmax_fp32", "tir.noalias": True})
    # 注意：必须用 ``sblock_alloc_buffer``（加到 root SBlock.alloc_buffers），
    # 而不是 ``alloc_buffer``（生成独立 AllocBuffer 语句节点）。
    # 否则 ``compute_at`` 内部的 ``IsOutputBlock`` 检查会用 root SBlock 的
    # ``alloc_buffers`` 列表，找不到这些中间 buffer 就会把 sum/exp/max 的
    # 写入误判成 output，从而抛 "is an output block"。
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


def _build_and_dump(mod, target, out_path: str, label: str) -> str:
    """跑 ``tvm.tirx.build`` 并把 c6678 源码落盘。"""
    runtime_mod = tvm.tirx.build(mod, target=target)
    print(f"[OK] {label} tvm.tirx.build returned: {runtime_mod}")
    try:
        source = runtime_mod.inspect_source("c6678")
    except Exception:  # pylint: disable=broad-except
        print(f"[WARN] {label} inspect_source('c6678') failed, fallback to default")
        source = runtime_mod.get_source()
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(source)
    print(f"[saved] {label} -> {out_path} ({len(source)} chars)")
    return source


def main() -> int:
    """端到端最小闭环：softmax PrimFunc → schedule → build → c6678 C 源码。"""
    mod = tvm.IRModule({"softmax_fp32": softmax_fp32})
    target = tvm.target.Target("c6678")
    print(f"[step] mod constructed, target = {target}")

    with target:
        scheduled_mod = dlight.ApplyDefaultSchedule(c6678_dlight.Softmax())(mod)

    is_scheduled = scheduled_mod["softmax_fp32"].attrs.get("tirx.is_scheduled")
    print(f"[S2-A] dlight.c6678.Softmax applied: tirx.is_scheduled={is_scheduled}")

    print("---- scheduled PrimFunc (TIR text) ----")
    print(scheduled_mod["softmax_fp32"].script())
    print("---- end TIR ----")

    try:
        scheduled_src = _build_and_dump(scheduled_mod, target, SCHEDULED_OUT, "scheduled")
    except Exception:  # pylint: disable=broad-except
        print("[FAIL] scheduled tvm.tirx.build raised:")
        traceback.print_exc()
        return 1

    baseline_mod = tvm.IRModule({"softmax_fp32": softmax_fp32})
    try:
        baseline_src = _build_and_dump(baseline_mod, target, BASELINE_OUT, "baseline")
    except Exception:  # pylint: disable=broad-except
        print("[FAIL] baseline tvm.tirx.build raised:")
        traceback.print_exc()
        return 1

    print(
        "[diff] sizes: scheduled=%d chars, baseline=%d chars, delta=%+d"
        % (len(scheduled_src), len(baseline_src), len(scheduled_src) - len(baseline_src))
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
