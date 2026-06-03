"""C6678 LSTM extern composite wrapper smoke test.

This is intentionally an extern-call integration test rather than a full LSTM
pattern matcher.  The current BSP-side LSTM implementation in ``examples.md``
is a composite kernel, so the first TVM-side step is to prove that a user-visible
PrimFunc can lower to a bare-C wrapper calling ``fp_lstm_s``.
"""

from __future__ import annotations

import tvm
from tvm.script import tirx as T


@T.prim_func
def lstm_fp32_extern(
    Output: T.Buffer((1024,), "float32"),
    Input: T.Buffer((1024,), "float32"),
    Params: T.Buffer((8,), "uint64"),
    core_mask: T.int32,
):
    """Forward to the BSP composite LSTM kernel: fp_lstm_s(output, input, params, core_mask)."""
    T.func_attr({"global_symbol": "lstm_fp32_extern", "tir.noalias": True})
    T.evaluate(
        T.call_extern(
            "",
            "fp_lstm_s",
            Output.data,
            Input.data,
            Params.data,
            core_mask,
        )
    )


def _build_and_get_source() -> str:
    mod = tvm.IRModule({"lstm_fp32_extern": lstm_fp32_extern})
    target = tvm.target.Target("c6678")
    runtime_mod = tvm.tirx.build(mod, target=target)
    return runtime_mod.inspect_source("c6678")


def test_lstm_extern_wrapper_compiles_to_c_source() -> None:
    src = _build_and_get_source()
    assert "void lstm_fp32_extern(float* Output, float* Input, uint64_t* Params, int32_t core_mask)" in src
    assert "fp_lstm_s(Output, Input, Params, core_mask)" in src
    assert "__tvm_ffi_lstm_fp32_extern" not in src
    print(f"[OK] LSTM extern c6678 wrapper generated ({len(src)} chars)")


if __name__ == "__main__":
    test_lstm_extern_wrapper_compiles_to_c_source()
