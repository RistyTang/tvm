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
"""C6678 极简 matmul schedule（路线图 §4.2 A.6 起点 + §4.2 A.5 proper 接口）。

该规则的目标是**先打通端到端最小闭环**：

1. 不做任何 cache_write / tensorize（A.6 之后再补）；
2. 当前 ``cache_read("global.l2") + compute_at(k_outer) + annotate``
   走 A.9 / A.10 step1 路径，配合 dlight ``MatmulGemmTemplate`` 完成具体改写；
3. 只在 ``target.kind.name == "c6678"`` 时返回 schedule，其它 target
   直接 ``None``。

A.5 proper 落地后，本 rule 退化为薄壳：

* 抽特征：调用 ``tvm.tirx.analysis.extract_features``；
* 选模板：调用 ``dlight.c6678.dispatcher.select_template``；
* 改 IR：交给 ``ScheduleTemplate.apply``。

历史上 rule 内嵌的 ``_pick_factor((32, 16, 8, 4, 2, 1))`` 已迁移到 A.4
``c6678_features._compute_tile_hint_and_alloc``，这里通过 ``feats.tile_hint``
透明使用。两者候选集严格一致，dispatcher 上线前后产出的 IR 与 C 源码应
完全相同（"零差分"约束，由 ``Test4dsp/tests/test_c6678_matmul_codegen.py``
+ ``probe_c6678_build_baseline.py`` 共同把守）。
"""

from __future__ import annotations

from tvm import s_tir, tirx
from tvm.target import Target

from ..analysis import normalize_prim_func
from ..base import try_inline_contiguous_spatial
from .base import C6678ScheduleRule
from .dispatcher import features_for_func, select_template


class Matmul(C6678ScheduleRule):
    """极简 matmul / GEMM schedule，匹配三层循环 ``[S, S, R]`` 的 reduce block。

    支持的 PrimFunc 形态（端到端 MVP 阶段足够）：

    .. code-block:: python

        @T.prim_func
        def matmul(A: T.Buffer((M, K), "float32"),
                   B: T.Buffer((K, N), "float32"),
                   C: T.Buffer((M, N), "float32")):
            for i, j, k in T.grid(M, N, K):
                with T.sblock("matmul"):
                    vi, vj, vk = T.axis.remap("SSR", [i, j, k])
                    with T.init():
                        C[vi, vj] = T.float32(0)
                    C[vi, vj] = C[vi, vj] + A[vi, vk] * B[vk, vj]
    """

    def apply(  # pylint: disable=too-many-locals,too-many-return-statements
        self,
        func: tirx.PrimFunc,
        target: Target,
        _: bool,
    ) -> None | s_tir.Schedule | list[s_tir.Schedule]:
        """对单个 PrimFunc 尝试套用 c6678 matmul schedule。

        Parameters
        ----------
        func : tirx.PrimFunc
            待 schedule 的 TIR PrimFunc。
        target : Target
            目标硬件描述，用于 ``is_target_available`` 守卫。
        _ : bool
            ``tunable`` 占位，本 MVP 阶段不区分。

        Returns
        -------
        sch : s_tir.Schedule | None
            匹配成功时返回完成 ``tile + reorder + cache_read + parallel`` 的
            Schedule；其它情况返回 ``None``，让 ``ApplyDefaultSchedule`` 跳过此函数。
        """
        if not isinstance(func, tirx.PrimFunc) or not self.is_target_available(target):
            return None

        # 1) 抽特征（A.4）
        feats = features_for_func(func, target)
        if feats is None or not feats.blocks:
            return None

        # 2) 选模板（A.5 proper）
        template = select_template(feats, block_idx=0)
        if template is None:
            return None

        # 3) 在已有 sch 上完成 normalize + try_inline 校验，确保 schedule 与
        #    features 抽取看到的是同一棵 IR（normalize_prim_func 是确定性的，
        #    不会因 sch 不同而产生不同结构）。
        sch = s_tir.Schedule(func)
        block_infos = normalize_prim_func(sch)
        if block_infos is None:
            return None
        block_infos = try_inline_contiguous_spatial(sch, block_infos)
        if block_infos is None or len(block_infos) != 1:
            return None
        block_info = block_infos[0]
        if block_info.dom_kind() != feats.blocks[0].dom_kind:
            # features 与 sch 看到的形态不一致，保守跳过
            return None

        # 4) 模板就地改写 sch（IR 改写责任全部下沉到模板）
        template.apply(sch, block_info, feats, block_idx=0)
        return sch

