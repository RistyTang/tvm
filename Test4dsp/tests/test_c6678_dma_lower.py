"""C6678DMALower failure-mode smoke tests."""

from __future__ import annotations

import tvm
from tvm.script import tirx as T
from tvm.tirx.transform import C6678DMALower


@T.prim_func
def _bad_dma_shape(
    A: T.Buffer((16,), "float32"),
    B: T.Buffer((16,), "float32"),
):
    T.func_attr({"target": T.target("c6678"), "global_symbol": "bad_dma_shape"})
    for i in T.serial(16):
        with T.sblock("stage"):
            T.sblock_attr({"c6678.dma_load": "load_row_major_tile"})
            vi = T.axis.spatial(16, i)
            B[vi] = A[vi]


def test_unlowered_dma_annotation_raises() -> None:
    """A 2D DMA annotation on a 1D loop must fail instead of being ignored."""
    mod = tvm.IRModule({"bad_dma_shape": _bad_dma_shape})
    raised = False
    try:
        C6678DMALower()(mod)
    except (ValueError, tvm.TVMError) as exc:
        raised = True
        msg = str(exc)
        assert "failed to lower all c6678.dma_load annotations" in msg
    assert raised, "C6678DMALower must reject unconsumed c6678.dma_load annotations"


if __name__ == "__main__":
    test_unlowered_dma_annotation_raises()
    print("[OK] C6678DMALower rejects unlowered DMA annotations")
