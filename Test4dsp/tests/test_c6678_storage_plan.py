"""临时单测脚本：验证 C6678StoragePlan pass 雏形（路线图 A.2）。"""

import tvm
from tvm.script import tirx as T


@T.prim_func
def f(A: T.Buffer((16, 16), "float32"),
      B: T.Buffer((16, 16), "float32"),
      C: T.Buffer((16, 16), "float32")):
    T.func_attr({"target": T.target("c6678"), "global_symbol": "f"})
    for i, j in T.grid(16, 16):
        with T.sblock("c"):
            vi, vj = T.axis.remap("SS", [i, j])
            C[vi, vj] = A[vi, vj] + B[vi, vj]


def main():
    mod = tvm.IRModule({"f": f})
    plan_pass = tvm.tirx.transform.C6678StoragePlan()
    mod2 = plan_pass(mod)
    plan = mod2["f"].attrs["c6678.storage_plan"]
    print("plan length:", len(plan))
    for entry in plan:
        print({str(k): (int(v) if hasattr(v, '__int__') and not isinstance(v, str) else str(v))
               for k, v in entry.items()})
    print("A.2 OK")


if __name__ == "__main__":
    main()
