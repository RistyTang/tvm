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
# pylint: disable=missing-function-docstring
"""C6678 elementwise schedule rules.

This module currently contains the first elementwise MVP: ``ElementGreaterEqual``.
It matches a single same-shape injective block whose store value is ``>=``,
tiles the outermost spatial loop, stages both input streams through L2 by
``dma_trans``, and lets ``C6678MulticoreLower`` turn the outer loop into the
bare-metal ``core_mask`` dispatch sequence.
"""

from __future__ import annotations

from tvm import s_tir, tirx
from tvm.target import Target
from tvm.tirx.c6678_config import from_target, l2_dma_block_elems

from ..analysis import normalize_prim_func
from ..base import try_inline_contiguous_spatial
from .base import C6678ScheduleRule


def _is_greater_equal_store(stmt: tirx.Stmt) -> bool:
    """Return True when ``stmt`` is ``Out[...] = lhs >= rhs``."""
    return isinstance(stmt, tirx.BufferStore) and isinstance(stmt.value, tirx.GE)


def _static_loop_extent(loop_stmt: tirx.For) -> int | None:
    """Return a Python int for a static loop extent, otherwise ``None``."""
    extent = loop_stmt.extent
    if isinstance(extent, tirx.IntImm):
        return int(extent.value)
    return None


class ElementGreaterEqual(C6678ScheduleRule):
    """Schedule a same-shape ``float32 >= float32 -> bool`` elementwise block.

    The first DMA-enabled version is intentionally conservative: it only stages
    the two input streams to L2 and writes the bool output directly to DDR.
    Output DMA store and broadcast/scalar variants are left to the next phase.
    """

    def apply(
        self,
        func: tirx.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None

        sch = s_tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) != 1:
            return None

        block_info = block_infos[0]
        if "R" in block_info.dom_kind() or block_info.is_reduction():
            return None

        block_stmt = sch.get(block_info.block_rv)
        if not _is_greater_equal_store(block_stmt.body):
            return None

        loops = sch.get_loops(block_info.block_rv)
        if not loops:
            return None

        cfg = from_target(target)
        outer_extent = _static_loop_extent(sch.get(loops[0]))
        block_elems = l2_dma_block_elems(
            "float32",
            cfg.l2_size,
            num_input_buffers=2,
            num_output_buffers=1,
            align_bytes=cfg.dma_align_bytes,
            max_elems=outer_extent,
        )

        outer, inner = sch.split(loops[0], factors=[None, block_elems])
        if len(loops) > 1:
            sch.reorder(outer, *loops[1:], inner)

        for read_index in range(2):
            l2_block = sch.cache_read(block_info.block_rv, read_index, "global.l2")
            sch.compute_at(l2_block, outer, preserve_unit_loops=False)
            sch.annotate(l2_block, "c6678.dma_load", "dma_trans")

        sch.parallel(outer)
        return sch
