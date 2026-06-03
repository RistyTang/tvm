# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=missing-function-docstring, invalid-name
"""C6678 极简 softmax schedule（路线图 §4.2 A.6 / PR-S2 phase A）。

设计目标
--------
本 rule 让用户写一份"教科书"形 softmax PrimFunc 之后，``tvm.tirx.build``
可以直接产出 c6678 bare-C 源码，匹配 ``Test4dsp/examples.md`` 的
``fp_softmax_p / fp_softmax_s`` 算法语义（数值稳定的三 pass：max → exp+sum →
div），但**不**做内层 inner 分块 + dma_trans staging（那部分留给 PR-S2 phase B
完成；本 rule 仅打通"PrimFunc → schedule → c6678 source"最小闭环）。

PrimFunc 形态契约
-----------------
本 rule 仅匹配下面的 4 块结构（dlight/cpu/reduction.py 风格）：

.. code-block:: python

    @T.prim_func
    def softmax(A: T.Buffer((O, I), "float32"),
                Out: T.Buffer((O, I), "float32")):
        T_max     = T.alloc_buffer((O,), "float32")
        T_exp     = T.alloc_buffer((O, I), "float32")
        T_sum     = T.alloc_buffer((O,), "float32")
        # block 0: SR  reduction max     (T_max[i] = max A[i, k])
        # block 1: SS  injective         (T_exp[i, k] = exp(A[i, k] - T_max[i]))
        # block 2: SR  reduction sum     (T_sum[i] = sum T_exp[i, k])
        # block 3: SS  injective         (Out[i, k] = T_exp[i, k] / T_sum[i])

匹配条件：

1. ``normalize_prim_func`` 返回**恰好 4 个 block**；
2. ``dom_kind`` 序列为 ``["SR", "SS", "SR", "SS"]``；
3. block-1 的 body 写入是 ``BufferStore(value=Call(tirx.exp, ...))``；
4. block-3 的 body 写入是 ``BufferStore(value=Div)``；
5. **静态 shape**：所有 dom_extent 必须是 IntImm（PR-S3 再放开）。

满足以上即触发 schedule。否则返回 None 让上游跳过。

Schedule 策略（PR-S2 phase A 极简版）
-------------------------------------
- 用最后 (epilogue) 块 ``Out[i, k] = T_exp[i, k] / T_sum[i]`` 的外层 spatial
  loop 作为 anchor；
- 把前面 3 块 ``compute_at`` 到该 spatial loop 下，构成"每行一次性算完
  max → exp+sum → div"的串行结构。这与 examples.md ``fp_softmax_p`` 结构等价；
- ``parallel(spatial)`` 给 A.8 多核 lower 留入口；
- 不做 vectorize（c6678 的 LLVM 后端不可用，C codegen 直接吐 serial loop）；
- 不做 cache_read / dma_trans（phase B）。

PR-S2 phase B（后续）
---------------------
- 沿 ``axis_size`` 切 ``BLOCK_CAPACITY = 115200`` floats 的 staging block，
  每个 staging block 注解 ``c6678.dma_load = "dma_trans"``，让 A.9
  ``C6678DMALower`` 自动 lower 成 ``dma_trans(src, dst, size_bytes)``；
- L2 暂存预算检查：input/output/sum 三段 alloc 不超过
  ``c6678_config.l2_size``；
- 这些都不影响 phase A 的 IR 结构 —— phase B 是给 phase A 的输出加 staging 装饰。
"""

from __future__ import annotations

import tvm
from tvm import s_tir, tirx
from tvm.target import Target
from tvm.tirx.expr import BufferLoad, Call, Cast

from ..analysis import normalize_prim_func
from .base import C6678ScheduleRule


_OP_EXP = tvm.ir.op.Op.get("tirx.exp")


def _store_value_is_div(stmt: tirx.Stmt) -> bool:
    """判定 block body 是 ``Out[...] = X / Y`` 形式。"""
    if not isinstance(stmt, tirx.BufferStore):
        return False
    return isinstance(stmt.value, tirx.expr.Div)


def _store_value_is_exp(stmt: tirx.Stmt) -> bool:
    """判定 block body 是 ``Out[...] = exp(...)`` 形式（允许 ``exp(A - max)``）。"""
    if not isinstance(stmt, tirx.BufferStore):
        return False
    val = stmt.value
    # 允许直接 cast 包一层（例如 fp16 输入提前 promote 到 fp32 再做 exp）。
    if isinstance(val, Cast):
        val = val.value
    if not isinstance(val, Call):
        return False
    return val.op == _OP_EXP


def _all_static(extents) -> bool:
    """所有 extent 都是静态整型时返回 True。

    ``SBlockInfo.dom()`` 返回 ``list[int | tirx.PrimExpr]``，因此既要兼容
    Python int，也要兼容 ``tirx.IntImm``。任意非整型 PrimExpr（含 Var）视
    为动态，PR-S2 phase A 一律拒绝以保留最简语义。
    """
    for e in extents:
        if isinstance(e, int):
            continue
        if isinstance(e, tirx.IntImm):
            continue
        return False
    return True


def _identify_softmax_blocks(sch: s_tir.Schedule, block_infos):
    """检测当前 PrimFunc 是否符合本 rule 约定的 4 块 softmax 形态。

    Returns
    -------
    matched : bool
        是否命中；命中后调用方可以直接套用 schedule。
    """
    if block_infos is None or len(block_infos) != 4:
        return False

    expected_kinds = ["SR", "SS", "SR", "SS"]
    for bi, kind in zip(block_infos, expected_kinds):
        if bi.dom_kind() != kind:
            return False
        if not _all_static(bi.dom()):
            return False

    # block 1 = exp（injective with exp call）；block 3 = div（injective with /）
    exp_block_stmt = sch.get(block_infos[1].block_rv)
    div_block_stmt = sch.get(block_infos[3].block_rv)
    if not _store_value_is_exp(exp_block_stmt.body):
        return False
    if not _store_value_is_div(div_block_stmt.body):
        return False

    # block 0 / block 2：reduction，应该是 SR + reduction_block 标志位
    if not block_infos[0].is_reduction():
        return False
    if not block_infos[2].is_reduction():
        return False
    return True


class Softmax(C6678ScheduleRule):
    """C6678 softmax schedule rule（PR-S2 phase A 极简版）。

    匹配 ``["SR", "SS", "SR", "SS"]`` 4 块、静态 shape、块-1 是 ``exp``、
    块-3 是除法的 PrimFunc，应用 ``parallel(outer) + compute_at`` 极简
    schedule，让 ``tvm.tirx.build`` 跑得通。
    """

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements
        self,
        func: tirx.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        """对单个 PrimFunc 尝试套用 c6678 softmax schedule。

        Parameters
        ----------
        func : tirx.PrimFunc
            待 schedule 的 PrimFunc，要求是 4 块 softmax IR。
        target : Target
            目标硬件描述，必须是 c6678 才会真正生效。
        _ : bool
            ``tunable`` 占位。

        Returns
        -------
        sch : s_tir.Schedule | None
            匹配成功时返回完成 ``parallel + compute_at`` 的 Schedule；
            其它情况返回 None，让 ``ApplyDefaultSchedule`` 跳过此函数。
        """
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        if not _identify_softmax_blocks(sch, block_infos):
            return None

        # 用 epilogue（块-3，除法块）的最外层 spatial loop 作 anchor。
        # SS 块的所有 loop 都是 spatial；这里取第一个（外层 row 维）。
        epilogue = block_infos[3]
        loops = sch.get_loops(epilogue.block_rv)
        if len(loops) < 2:
            return None
        outer = loops[0]

        # 调度顺序（与 dlight/cpu/reduction.py 保持一致）：
        # 1) 先在 epilogue (div) 的外层 spatial loop 上 ``parallel``；
        # 2) 再把前 3 块逆序 ``compute_at`` 到该 loop 下。
        # parallel 先做的目的：让外层 loop 在 compute_at 时已经稳定为 thread-binding /
        # parallel kind，下游的 IsOutputBlock 检查会用整个 root scope 作为 scope_root，
        # 从而正确把 ``T_sum`` 等 alloc_buffer 视为 scope-内分配（非输出）。
        sch.parallel(outer)

        # 逆序保证依赖：先 sum（最贴近 div），再 exp，最后 max。
        for bi in reversed(block_infos[:3]):
            sch.compute_at(bi.block_rv, outer, preserve_unit_loops=True)
        return sch
