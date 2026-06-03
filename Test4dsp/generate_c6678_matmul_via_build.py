"""C6678 端到端最小闭环 demo（路线图 §4.2 A.5/A.6 起步验证）。

用户视角：

1. 用 `@T.prim_func` 写一个 fp32 128x128x128 matmul（全部默认 scope，落在 DDR）；
2. `tvm.target.Target("c6678")` 拿到目标；
3. 通过 `dlight.ApplyDefaultSchedule(c6678.Matmul())` 套一层极简专家 schedule；
4. `tvm.tirx.build(mod, target=...)` 完成 IR→TIR→codegen，吐出 6678 C 源码；
5. 顺带打印 A.2 `C6678StoragePlan` 在 attrs 上挂的 storage plan，证明 c6678 专属
   pass 已挂进 `default_s_tir_pipeline`（虽然其它 pass 后续会丢掉这个 attr，
   这里先单独跑一次让用户能看到结果）；
6. 同时把"套了 schedule 的版本"和"完全没 schedule 的 baseline 版本"两份完整
   C 源码 dump 到 ``Test4dsp/tests/`` 下，方便对比 codegen 差异。

后续 A.3 ~ A.6 的所有改动都不应改变本 demo 的接口形态，只会让生成的源码更优。
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

M, N, K = 128, 128, 128

# 落盘路径：与 generate_c6678_matmul.py 保持一致放在 Test4dsp/tests/ 下。
_OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests")
SCHEDULED_OUT = os.path.join(_OUT_DIR, "generated_c6678_matmul_via_build.c")
BASELINE_OUT = os.path.join(_OUT_DIR, "generated_c6678_matmul_baseline.c")


@T.prim_func
def matmul_fp32(
    A: T.Buffer((M, K), "float32"),
    B: T.Buffer((K, N), "float32"),
    C: T.Buffer((M, N), "float32"),
):
    """fp32 matmul：C = A @ B，shape 128x128x128，全部走默认 scope (DDR)。"""
    T.func_attr({"global_symbol": "matmul_fp32", "tir.noalias": True})
    for i, j, k in T.grid(M, N, K):
        with T.sblock("matmul"):
            vi, vj, vk = T.axis.remap("SSR", [i, j, k])
            with T.init():
                C[vi, vj] = T.float32(0)
            C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]


def _print_storage_plan(mod: tvm.IRModule, target: tvm.target.Target) -> None:
    """单独跑 BindTarget + C6678StoragePlan，观察 storage_plan 在 attrs 上的形态。

    为什么要单独跑：`default_s_tir_pipeline` 的后续 pass（特别是 lower 阶段的
    AttrStmt 处理与 host/device 拆分）会把 `c6678.storage_plan` 这个 PrimFunc
    属性当作"未消费"的元数据，最终输出阶段不一定保留。本函数只为可视化、不影响
    主链路。
    """
    bound = tvm.tirx.transform.BindTarget(target)(mod)
    bound = tvm.tirx.transform.C6678StoragePlan()(bound)
    plan = bound["matmul_fp32"].attrs.get("c6678.storage_plan")
    if plan is None:
        print("[A.2] storage_plan: <None>")
        return
    print(f"[A.2] storage_plan entries: {len(plan)}")
    for entry in plan:
        normalized = {}
        for k, v in entry.items():
            key = str(k)
            if hasattr(v, "__int__") and not isinstance(v, str):
                normalized[key] = int(v)
            else:
                normalized[key] = str(v)
        print("       ", normalized)


def _build_and_dump(mod: tvm.IRModule, target: tvm.target.Target, out_path: str, label: str) -> str:
    """执行 ``tvm.tirx.build`` 并把生成的完整 C 源码写到 ``out_path``。

    Parameters
    ----------
    mod : tvm.IRModule
        待 build 的 IR 模块（可能套了 schedule，也可能没套）。
    target : tvm.target.Target
        目标硬件描述，用于路由到 ``target.build.c6678``。
    out_path : str
        生成 C 源码的落盘路径。
    label : str
        日志前缀，方便区分两次 build。

    Returns
    -------
    source : str
        生成的完整 C 源码，调用方可继续对其计数 / 摘要。
    """
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
    """端到端最小闭环：matmul PrimFunc → schedule → tvm.tirx.build → C 源码。"""
    mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
    target = tvm.target.Target("c6678")
    print(f"[step] mod constructed, target = {target}")

    # 单独跑一次 A.2 让用户看到 storage plan，不参与主 build。
    _print_storage_plan(mod, target)

    # ApplyDefaultSchedule 需要一个 target context 让 ScheduleRule._get_target 拿到 target；
    # 同时也方便后续 dispatcher 内部 Target.current() 兜底。
    with target:
        scheduled_mod = dlight.ApplyDefaultSchedule(c6678_dlight.Matmul())(mod)

    is_scheduled = scheduled_mod["matmul_fp32"].attrs.get("tirx.is_scheduled")
    print(f"[A.6] dlight.c6678.Matmul applied: tirx.is_scheduled={is_scheduled}")

    print("---- scheduled PrimFunc (TIR text) ----")
    print(scheduled_mod["matmul_fp32"].script())
    print("---- end TIR ----")

    try:
        scheduled_src = _build_and_dump(scheduled_mod, target, SCHEDULED_OUT, "scheduled")
    except Exception:  # pylint: disable=broad-except
        print("[FAIL] scheduled tvm.tirx.build raised:")
        traceback.print_exc()
        return 1

    # baseline：不套任何 schedule，直接 build。新建一份 IRModule 避免被前面的 pipeline 状态污染。
    baseline_mod = tvm.IRModule({"matmul_fp32": matmul_fp32})
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
