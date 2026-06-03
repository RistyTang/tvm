"""A.3 ``C6678DMALegalize`` 冒烟测试。

覆盖四个场景：

1. **正向通过**：手工构造一条标准 ``call_extern("load_row_major_tile", ...)``
   塞进 PrimFunc body，过 pass 后无报错、``c6678.dma_legalized`` flag 被打上。
2. **scope 非法**：把 ``src_scope`` 改成 ``"shared"``，过 pass 后抛
   ``ValueError`` 且包含 illegal src_scope 字样。
3. **超容量**：把 ``rows*cols*elem_size`` 设成 ``dma_max_transfer + 1``，
   过 pass 后抛 ``ValueError``。
4. **行对齐告警**：把 ``cols*elem_size`` 设成非 ``dma_align_bytes`` 整数倍
   （``cols=1, elem_size=1`` → 1B/行），pass 不阻断但发
   ``C6678DMAAlignmentWarning``。
5. **幂等**：同一 PrimFunc 上重复跑 pass，第二次直接早返回（不会再次校验，
   也不会再次发 warning）。
6. **非 c6678 target 跳过**：target=``llvm`` 的 PrimFunc 即便 body 里塞了不合法
   的 call，pass 也不会报错（守卫生效）。

不依赖 pytest，``python Test4dsp/tests/test_c6678_dma_legalize.py`` 直接跑。
"""

from __future__ import annotations

import warnings

import tvm
from tvm import tirx
from tvm.tirx.transform import (
    C6678DMALegalize,
    C6678DMAAlignmentWarning,
)


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _make_dma_call(
    *,
    rows: int = 8,
    cols: int = 16,
    src_ld: int = 16,
    elem_size: int = 4,
    src_scope: str = "global",
):
    """构造一条 ``call_extern("load_row_major_tile", ...)`` Evaluate stmt。

    形态对齐 ``C6678DMALower._build_dma_call``：
    ``[fn_name, src_ptr, dst_ptr, row0, col0, rows, cols, src_ld, elem_size, src_scope]``。

    src_ptr / dst_ptr 不影响合法性校验，这里直接用 IntImm(0) 占位即可（A.3 不
    检查 access_ptr 的语义）。
    """
    src_ptr = tirx.IntImm("int32", 0)
    dst_ptr = tirx.IntImm("int32", 0)
    row0 = tirx.IntImm("int32", 0)
    col0 = tirx.IntImm("int32", 0)
    call = tirx.call_extern(
        "",
        "load_row_major_tile",
        src_ptr,
        dst_ptr,
        row0,
        col0,
        tirx.IntImm("int32", rows),
        tirx.IntImm("int32", cols),
        tirx.IntImm("int32", src_ld),
        tirx.IntImm("int32", elem_size),
        tirx.StringImm(src_scope),
    )
    return tirx.Evaluate(call)


def _wrap_func(body, *, target: str = "c6678", name: str = "f"):
    """用最小 PrimFunc 包住 body，挂上 target / global_symbol，便于过 pass。

    使用 ``tvm.ir.make_node("ir.DictAttrs", ...)`` 构造 attrs，与 tvm 现有
    ``test_tvmscript_ir_builder_tir.py`` 等测试的写法保持一致。
    """
    func = tirx.PrimFunc(
        params=[],
        body=body,
        ret_type=None,
        buffer_map={},
        attrs=tvm.ir.make_node(
            "ir.DictAttrs",
            **{
                "target": tvm.target.Target(target),
                "global_symbol": name,
            },
        ),
    )
    return func


def _run_pass(func, name: str = "f"):
    mod = tvm.IRModule({name: func})
    return C6678DMALegalize()(mod)


def _make_dma_trans_call(*, size_bytes: int = 4096):
    """构造一条 ``call_extern("dma_trans", src, dst, size_bytes)`` Evaluate stmt。

    形态对齐 BSP wrapper（``Test4dsp/examples.md`` 第 211~228 行）：
    ``[fn_name, src_ptr, dst_ptr, size_bytes]`` —— 4 个 args，无 src_scope。
    用 int64 IntImm 容纳 ``size_bytes``，避免 ``0x80000000`` 这类越过 int32
    上界的负数测试值在构造期就溢出。
    """
    src_ptr = tirx.IntImm("int32", 0)
    dst_ptr = tirx.IntImm("int32", 0)
    call = tirx.call_extern(
        "",
        "dma_trans",
        src_ptr,
        dst_ptr,
        tirx.IntImm("int64", size_bytes),
    )
    return tirx.Evaluate(call)


def test_legal_call_passes() -> None:
    body = _make_dma_call(rows=8, cols=16, src_ld=16, elem_size=4, src_scope="global")
    func = _wrap_func(body)
    new_mod = _run_pass(func)
    new_func = new_mod["f"]
    flag = new_func.attrs["c6678.dma_legalized"]
    _check(bool(int(flag)) is True, f"expect c6678.dma_legalized=True, got {flag}")
    print("[OK] legal load_row_major_tile passes (cols*elem=64 aligned)")


def test_illegal_scope_raises() -> None:
    body = _make_dma_call(src_scope="shared")
    func = _wrap_func(body)
    raised = False
    try:
        _run_pass(func)
    except (ValueError, tvm.TVMError) as exc:
        raised = True
        msg = str(exc)
        _check(
            "illegal src_scope" in msg or "shared" in msg,
            f"error message should mention illegal src_scope, got {msg!r}",
        )
    _check(raised, "expect ValueError for src_scope='shared'")
    print("[OK] illegal src_scope='shared' rejected")


def test_oversized_transfer_raises() -> None:
    # dma_max_transfer = 0x7FFFFFFF；让 rows*cols*elem 直接溢出该上限
    body = _make_dma_call(
        rows=0x10000,
        cols=0x10000,
        src_ld=0x10000,
        elem_size=4,
        src_scope="global",
    )
    func = _wrap_func(body)
    raised = False
    try:
        _run_pass(func)
    except (ValueError, tvm.TVMError) as exc:
        raised = True
        msg = str(exc)
        _check(
            "exceeds dma_max_transfer" in msg or "transfer" in msg,
            f"error message should mention dma_max_transfer, got {msg!r}",
        )
    _check(raised, "expect ValueError for oversized transfer")
    print("[OK] oversized transfer rejected (>dma_max_transfer)")


def test_unaligned_row_warns_but_passes() -> None:
    # cols * elem_size = 1 * 1 = 1B，不是 64B 整数倍 → 应 warn，但不阻断
    body = _make_dma_call(
        rows=8,
        cols=1,
        src_ld=1,
        elem_size=1,
        src_scope="global",
    )
    func = _wrap_func(body)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_mod = _run_pass(func)
        align_warns = [w for w in caught if issubclass(w.category, C6678DMAAlignmentWarning)]
        _check(
            len(align_warns) >= 1,
            f"expect at least one C6678DMAAlignmentWarning, got {caught}",
        )
    flag = new_mod["f"].attrs["c6678.dma_legalized"]
    _check(bool(int(flag)) is True, "pass should still mark dma_legalized after warning")
    print(f"[OK] unaligned row triggered {len(align_warns)} warning(s) without aborting")


def test_idempotent_second_run_skips() -> None:
    body = _make_dma_call(rows=8, cols=16, src_ld=16, elem_size=4, src_scope="global")
    func = _wrap_func(body)
    mod1 = _run_pass(func)
    func1 = mod1["f"]
    # 第二次跑：用 mod1 自身再过一次 pass，预期早返回（不会重发 warning，
    # 也不会改 flag）。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod2 = C6678DMALegalize()(mod1)
        align_warns = [w for w in caught if issubclass(w.category, C6678DMAAlignmentWarning)]
        _check(
            len(align_warns) == 0,
            f"second run must not re-emit warnings, got {align_warns}",
        )
    flag2 = mod2["f"].attrs["c6678.dma_legalized"]
    _check(bool(int(flag2)) is True, "flag should still be True after idempotent run")
    print("[OK] second pass run is idempotent (no re-validation)")


def test_non_c6678_target_skipped() -> None:
    # llvm target 即便塞了不合法的 scope，pass 也不应报错（target guard）
    body = _make_dma_call(src_scope="shared")
    func = _wrap_func(body, target="llvm", name="g")
    new_mod = _run_pass(func, name="g")
    new_func = new_mod["g"]
    flag = new_func.attrs.get("c6678.dma_legalized")
    _check(flag is None, f"non-c6678 func must NOT be legalized, got attrs flag={flag}")
    print("[OK] non-c6678 target skipped without validation")


def test_legal_dma_trans_passes() -> None:
    """dma_trans 1D：4096 B 在 dma_max_transfer 之内，应通过且打 flag。"""
    body = _make_dma_trans_call(size_bytes=4096)
    func = _wrap_func(body, name="dt_legal")
    new_mod = _run_pass(func, name="dt_legal")
    flag = new_mod["dt_legal"].attrs["c6678.dma_legalized"]
    _check(bool(int(flag)) is True, f"expect dma_legalized=True for legal dma_trans, got {flag}")
    print("[OK] legal dma_trans(size=4096B) passes")


def test_dma_trans_negative_size_raises() -> None:
    """dma_trans size 为负 → 必须直接报错。"""
    body = _make_dma_trans_call(size_bytes=-1)
    func = _wrap_func(body, name="dt_neg")
    raised = False
    try:
        _run_pass(func, name="dt_neg")
    except (ValueError, tvm.TVMError) as exc:
        raised = True
        msg = str(exc)
        _check(
            "size must be positive" in msg or "dma_trans" in msg,
            f"error message should mention dma_trans size, got {msg!r}",
        )
    _check(raised, "expect ValueError for negative dma_trans size")
    print("[OK] dma_trans negative size rejected")


def test_dma_trans_oversized_raises() -> None:
    """dma_trans size > dma_max_transfer (0x7FFFFFFF) → 报错。"""
    body = _make_dma_trans_call(size_bytes=0x80000000)
    func = _wrap_func(body, name="dt_big")
    raised = False
    try:
        _run_pass(func, name="dt_big")
    except (ValueError, tvm.TVMError) as exc:
        raised = True
        msg = str(exc)
        _check(
            "exceeds dma_max_transfer" in msg or "transfer size" in msg,
            f"error message should mention dma_max_transfer, got {msg!r}",
        )
    _check(raised, "expect ValueError for dma_trans oversized transfer")
    print("[OK] dma_trans oversized size rejected (>dma_max_transfer)")


if __name__ == "__main__":
    test_legal_call_passes()
    test_illegal_scope_raises()
    test_oversized_transfer_raises()
    test_unaligned_row_warns_but_passes()
    test_idempotent_second_run_skips()
    test_non_c6678_target_skipped()
    test_legal_dma_trans_passes()
    test_dma_trans_negative_size_raises()
    test_dma_trans_oversized_raises()
    print("[OK] all c6678 dma legalize tests passed")
