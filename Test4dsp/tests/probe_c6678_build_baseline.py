"""Baseline 探测脚本：确认 tvm.tirx.build(target='c6678') 当前能否走通。

只用最朴素的 PrimFunc（fp32 128x128x128 matmul，全部默认 scope，没有任何 schedule），
看现状是"已经能产出 C 源码"还是"在某个 pass 处崩溃"。
"""

from __future__ import annotations

import sys
import traceback

import tvm
from tvm.script import tirx as T


@T.prim_func
def matmul_fp32(
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


def main() -> int:
    mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
    print("[step] mod constructed")

    # 先单独验证 A.2：BindTarget + 预跑 C6678StoragePlan，看 plan 是否会写入 attrs。
    target = tvm.target.Target("c6678")
    bound_mod = tvm.tirx.transform.BindTarget(target)(mod)
    bound_mod = tvm.tirx.transform.C6678StoragePlan()(bound_mod)
    plan = bound_mod["matmul_fp32"].attrs.get("c6678.storage_plan")
    print("[A.2] storage_plan entries:", None if plan is None else len(plan))
    if plan is not None:
        for entry in plan:
            print("       ", {str(k): (int(v) if hasattr(v, '__int__') and not isinstance(v, str) else str(v)) for k, v in entry.items()})

    try:
        runtime_mod = tvm.tirx.build(mod, target="c6678")
    except Exception as exc:
        print("[FAIL] tvm.tirx.build raised:")
        traceback.print_exc()
        return 1

    print("[OK] tvm.tirx.build returned:", runtime_mod)
    try:
        source = runtime_mod.inspect_source("c6678")
    except Exception as exc:
        print("[WARN] inspect_source('c6678') failed, fallback to default")
        source = runtime_mod.get_source()
    print("---- generated source (first 2000 chars) ----")
    print(source[:2000])
    print("---- end ----")
    return 0


if __name__ == "__main__":
    sys.exit(main())
