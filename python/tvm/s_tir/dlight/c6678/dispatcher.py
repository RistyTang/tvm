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
"""C6678 schedule 模板派发器（路线图 §4.2 A.5 proper）。

本模块完成"特征 → schedule 模板"的派发：上游 schedule rule（如
``dlight.c6678.Matmul``）拿到 PrimFunc 与 Target 之后，先调用
``tvm.tirx.analysis.extract_features`` 抽出
``C6678PrimFuncFeatures``，再交给 ``select_template`` 选模板，最终由模板
负责完成 ``cache_read / compute_at / annotate / parallel`` 等具体 schedule
动作。

关键设计
--------
* **模板与 schedule rule 解耦**：每种 (op_kind, dom_kind) 组合对应一个
  ``ScheduleTemplate`` 子类；新增模板只需在 ``_TEMPLATE_REGISTRY`` 里
  补一行，无需改 rule 入口。
* **L2 容量门禁**：派发器读 ``feats.config.l2_size``，再用模板自报的
  ``estimate_l2_bytes`` 与之比较；超量时模板返回 ``None`` 让 rule 走
  fallback（当前 fallback = "返回 None，跳过 c6678 schedule"）。
* **不接管 IR 改写**：派发器自身只挑模板、传 features，不直接 ``sch.xxx``；
  IR 改写全部在模板的 ``apply()`` 里完成 —— 这与 §4.3 选址原则一致。

可观测产物：dispatcher 落地后，``Test4dsp/generate_c6678_matmul_via_build.py``
的输出 char 数应保持不变（A.4 的 tile_hint 与原硬编码 (32,32,32) 在
``128x128x128`` matmul 上一致）；当 dom_extent 含非 32 倍数时，新派发器
会自动走 16/8/4/2/1 退化路径，避免老代码 ``return None`` 直接跳过。
"""

from __future__ import annotations

from typing import Optional

from tvm import s_tir, tirx
from tvm.target import Target
from tvm.tirx.analysis import C6678PrimFuncFeatures, extract_features


class ScheduleTemplate:
    """schedule 模板抽象基类。

    每个具体模板需要实现：
    * ``can_apply(feats, block_idx)`` —— 仅看 features 决定是否匹配该 block；
    * ``estimate_l2_bytes(feats, block_idx)`` —— 静态估算 L2 占用，让派发器
      做容量门禁；
    * ``apply(sch, block_info, feats, block_idx)`` —— 在已构造好的 Schedule
      上完成具体改写，返回 None 表示成功（沿用 ``s_tir.Schedule`` 的就地改
      写约定，schedule 句柄由 rule 持有）。

    说明：模板只负责一种 (op_kind, dom_kind) 组合；若 ``can_apply`` 返回
    False，dispatcher 会把它跳过去尝试下一个候选模板。
    """

    name: str = "<abstract>"

    def can_apply(
        self, feats: C6678PrimFuncFeatures, block_idx: int
    ) -> bool:  # pragma: no cover - abstract
        raise NotImplementedError

    def estimate_l2_bytes(
        self, feats: C6678PrimFuncFeatures, block_idx: int
    ) -> Optional[int]:  # pragma: no cover - abstract
        raise NotImplementedError

    def apply(
        self,
        sch: s_tir.Schedule,
        block_info,
        feats: C6678PrimFuncFeatures,
        block_idx: int,
    ) -> None:  # pragma: no cover - abstract
        raise NotImplementedError


class MatmulGemmTemplate(ScheduleTemplate):
    """``[S, S, R]`` 三层 reduce-style matmul 的默认模板。

    与原 ``dlight.c6678.Matmul.apply`` 保持等价，区别仅在 tile 大小来源：
    * 旧：硬编码 ``_pick_factor((32, 16, 8, 4, 2, 1))``；
    * 新：``feats.blocks[block_idx].tile_hint``，由 A.4 在 features 抽取
      阶段统一计算（候选集与原硬编码一致）。

    这意味着同一份 PrimFunc 在 dispatcher 上线前后产出的 IR / C 源码应
    完全一致 —— 这是验证 A.5 没有引入回归的"零差分"约束。
    """

    name: str = "matmul_gemm"

    def can_apply(self, feats: C6678PrimFuncFeatures, block_idx: int) -> bool:
        op = feats.blocks[block_idx]
        # 仅匹配静态三维 SSR 的 gemm；其它 (op_kind, dom_kind) 组合留给后续模板
        if op.op_kind != "gemm":
            return False
        if op.dom_kind != "SSR":
            return False
        if not op.is_static_shape:
            return False
        if len(op.dom_extents) != 3 or len(op.tile_hint) != 3:
            return False
        if any(t is None for t in op.tile_hint):
            return False
        return True

    def estimate_l2_bytes(
        self, feats: C6678PrimFuncFeatures, block_idx: int
    ) -> Optional[int]:
        return feats.blocks[block_idx].static_alloc_l2_bytes

    def apply(
        self,
        sch: s_tir.Schedule,
        block_info,
        feats: C6678PrimFuncFeatures,
        block_idx: int,
    ) -> None:
        op = feats.blocks[block_idx]
        ti, tj, tk = op.tile_hint  # type: ignore[misc]
        assert ti is not None and tj is not None and tk is not None

        loops = sch.get_loops(block_info.block_rv)
        if len(loops) != 3:
            raise RuntimeError(
                f"MatmulGemmTemplate expects 3 loops, got {len(loops)}"
            )
        i_loop, j_loop, k_loop = loops

        i_outer, i_inner = sch.split(i_loop, factors=[None, ti])
        j_outer, j_inner = sch.split(j_loop, factors=[None, tj])
        k_outer, k_inner = sch.split(k_loop, factors=[None, tk])

        sch.reorder(i_outer, j_outer, k_outer, i_inner, j_inner, k_inner)

        # A.10 step1：cache_read 两路输入到 ``global.l2`` + compute_at(k_outer)
        # + annotate(c6678.dma_load=...)，构成 A.9 ``C6678DMALower`` 的输入契约。
        # 详细约束见 ``Test4dsp/learning.md §4.8.3`` 第 3 条。
        a_l2 = sch.cache_read(block_info.block_rv, 0, "global.l2")
        b_l2 = sch.cache_read(block_info.block_rv, 1, "global.l2")
        sch.compute_at(a_l2, k_outer, preserve_unit_loops=True)
        sch.compute_at(b_l2, k_outer, preserve_unit_loops=True)
        for stage_blk, src_name in ((a_l2, "A"), (b_l2, "B")):
            sch.annotate(stage_blk, ann_key="c6678.dma_load", ann_val="load_row_major_tile")
            sch.annotate(stage_blk, ann_key="c6678.src_buffer", ann_val=src_name)
            sch.annotate(stage_blk, ann_key="c6678.src_scope", ann_val="global")

        sch.parallel(i_outer)
        sch.annotate(k_inner, ann_key="pragma_auto_unroll_max_step", ann_val=tk)
        sch.annotate(k_inner, ann_key="pragma_unroll_explicit", ann_val=1)


_TEMPLATE_REGISTRY: tuple[ScheduleTemplate, ...] = (
    MatmulGemmTemplate(),
)


def select_template(
    feats: C6678PrimFuncFeatures, block_idx: int = 0
) -> Optional[ScheduleTemplate]:
    """从 ``_TEMPLATE_REGISTRY`` 中挑出第一个 ``can_apply == True`` 且 L2 占用
    不超过 ``feats.config.l2_size`` 的模板。

    Parameters
    ----------
    feats : C6678PrimFuncFeatures
        由 ``tvm.tirx.analysis.extract_features`` 抽出的特征汇总。
    block_idx : int
        当前关心的 block 下标（MVP 阶段 PrimFunc 只剩一个主 block）。

    Returns
    -------
    template : ScheduleTemplate | None
        命中则返回模板实例；否则 ``None`` 让上游 rule 走 fallback。
    """
    if feats is None:
        return None
    if block_idx >= len(feats.blocks):
        return None
    l2_cap = feats.config.l2_size if feats.config is not None else None
    for tpl in _TEMPLATE_REGISTRY:
        if not tpl.can_apply(feats, block_idx):
            continue
        if l2_cap is not None:
            est = tpl.estimate_l2_bytes(feats, block_idx)
            if est is not None and est > l2_cap:
                continue
        return tpl
    return None


def features_for_func(
    func: tirx.PrimFunc,
    target: Target,
    func_name: str = "main",
) -> Optional[C6678PrimFuncFeatures]:
    """便捷封装：把 ``extract_features`` 重新导出，让 dlight 子模块不必直接
    依赖 ``tvm.tirx.analysis``。"""
    return extract_features(func, target, func_name=func_name)


__all__ = [
    "ScheduleTemplate",
    "MatmulGemmTemplate",
    "select_template",
    "features_for_func",
]
