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
"""C6678 多核派发降级 pass（路线图 §4.8 A.8）。

为什么需要这个 pass：``dlight.c6678.Matmul`` 在外层 spatial loop 上做了
``sch.parallel(i_outer)``，最终在 IR 里表现为 ``For(kind=ForKind.PARALLEL)``，
但 c6678 codegen 当前把 PARALLEL 当 SERIAL 输出（见
``codegen_c6678.cc::VisitStmt_(ForNode*)``），实际板上不会做多核派发。

本 pass 在 A.7 之后、``MakePackedAPI`` 之前执行，对 c6678 host PrimFunc 做：

1. **追加 ``core_mask`` 形参**：在 ``params`` 末尾追加一个 ``int32`` 标量 Var，
   函数签名从 ``void matmul_fp32(float* A, float* B, float* C)`` 变为
   ``void matmul_fp32(float* A, float* B, float* C, int core_mask)``，与
   ``Test4dsp/tests/generated_c6678_matmul.c`` 里的 ``fp_matmul_fusion_s`` 入口一致。
2. **替换最外层 ``ForKind.PARALLEL`` 循环**：原始形态：

   .. code-block:: python

       for ax0_0 in T.parallel(0, EXTENT):
           <body>

   改写为：

   .. code-block:: c

       int core_num = GetCoreNum(core_mask);
       int core_id  = GetLogicCoreId(core_mask, DNUM);
       if (core_num > 0 && core_id >= 0) {
           int rows_per_core = EXTENT / core_num;
           int start = core_id * rows_per_core;
           int end   = (core_id == core_num - 1) ? EXTENT : (start + rows_per_core);
           for (ax0_0 = start; ax0_0 < end; ++ax0_0) {  // ForKind.SERIAL
               <body>
           }
       }
       C6678E_SyncN(core_num, core_id);

   ``GetCoreNum / GetLogicCoreId / C6678E_SyncN`` 通过 ``tvm.tirx.call_extern``
   表达。其中 ``DNUM`` 作为硬件已知宏，由 `codegen_c6678` 识别后直接原样输出，
   不需要新增函数形参。
3. **守卫**：仅对 ``target.kind.name == "c6678"`` 的 PrimFunc 处理；幂等通过
   attrs ``"c6678.multicore_lowered" == True`` 标记防止重复执行；body 中没有
   ``ForKind.PARALLEL`` 时原样跳过。

落地位置：``python/tvm/s_tir/pipeline.py::_c6678_pre_packed_api_passes`` 中，
**A.7 之后、``MakePackedAPI()`` 之前**。

后续 A.9 ``C6678DMALower`` 需要的 ``core_id`` 表达式可以通过 attrs
``"c6678.multicore_lowered"`` 判断本 pass 是否已经把 ``core_id`` LetStmt 注入。
"""

from __future__ import annotations

import tvm
from tvm import tirx

from .function_pass import prim_func_pass


_CORE_MASK_NAME = "core_mask"
_CORE_NUM_NAME = "core_num"
_CORE_ID_NAME = "core_id"


def _is_c6678_func(func) -> bool:
    """是否是带 ``target.kind.name == "c6678"`` 的 PrimFunc。"""
    if func.attrs is None:
        return False
    target = func.attrs.get("target")
    if target is None:
        return False
    return target.kind.name == "c6678"


def _already_lowered(func) -> bool:
    """通过 attrs 标记保证幂等：避免 pass 在意外重跑时二次改写。"""
    if func.attrs is None:
        return False
    flag = func.attrs.get("c6678.multicore_lowered")
    if flag is None:
        return False
    return bool(int(flag))


def _has_parallel_for(stmt) -> bool:
    """快速判定 PrimFunc body 中是否存在 ``ForKind.PARALLEL`` 的循环。"""
    found = [False]

    def _visit(node):
        if isinstance(node, tirx.For) and int(node.kind) == int(tirx.ForKind.PARALLEL):
            found[0] = True

    tirx.stmt_functor.post_order_visit(stmt, _visit)
    return found[0]


def _build_multicore_body(parallel_for, core_mask_var):
    """根据原始 ``parallel`` 循环和 ``core_mask`` 形参构造多核派发 body。

    Parameters
    ----------
    parallel_for : tvm.tirx.For
        原始的 ``For(kind=PARALLEL, loop_var=L, min=0, extent=EXTENT, body=...)``
        循环。本函数会基于它生成 SERIAL 子循环。
    core_mask_var : tvm.tirx.Var
        函数入口新追加的 ``core_mask`` 形参 Var。

    Returns
    -------
    tvm.tirx.SeqStmt
        ``SeqStmt([if guard { serial loop }, Evaluate(C6678E_SyncN)])`` 形态，
        对应 BSP 中的多核派发模式。

    实现注意
    --------
    ``tirx`` 没有 ``LetStmt`` 形态（只有 PrimExpr 级别的 ``Let``，无法跨多语句
    复用），因此本函数直接把 ``GetCoreNum / GetLogicCoreId`` 的 call_extern
    内联到循环边界、guard 条件以及 SyncN 调用三个位置。``GetCoreNum /
    GetLogicCoreId`` 是 BSP 纯函数，重复调用对功能与性能均无影响（codegen 后会
    形如 ``GetCoreNum(core_mask) > 0`` 这种字面量表达式）。后续 A.9 / A.10
    若需要 ``core_id`` 表达式，应通过 attrs ``c6678.multicore_lowered`` 判定
    并自行重新构造 call_extern，与本 pass 解耦。
    """
    loop_var = parallel_for.loop_var
    extent = parallel_for.extent
    orig_body = parallel_for.body

    # call_extern("int32", "GetCoreNum", core_mask) -> 与 BSP 一致。
    def _core_num():
        return tirx.call_extern("int32", "GetCoreNum", core_mask_var)

    dnum_var = tirx.Var("DNUM", "int32")

    # call_extern("int32", "GetLogicCoreId", core_mask, DNUM) -> 与用户约定一致。
    def _core_id():
        return tirx.call_extern("int32", "GetLogicCoreId", core_mask_var, dnum_var)

    # 切片 [start, end)，与 generated_c6678_matmul.c 完全一致：
    #   rows_per_core = EXTENT / core_num
    #   start = core_id * rows_per_core
    #   end   = (core_id == core_num - 1) ? EXTENT : (start + rows_per_core)
    rows_per_core = tirx.FloorDiv(extent, _core_num())
    start = _core_id() * rows_per_core
    end = tirx.Select(
        _core_id() == _core_num() - 1,
        extent,
        start + rows_per_core,
    )

    # 重新构造 SERIAL 循环：``for (loop_var = start; loop_var < end; ++loop_var)``。
    # tirx.For 的 ``min`` 即起始值、``extent`` 即长度，所以 length = end - start。
    new_serial = tirx.For(
        loop_var=loop_var,
        min=start,
        extent=end - start,
        kind=tirx.ForKind.SERIAL,
        body=orig_body,
    )

    # if (core_num > 0 and core_id >= 0) { new_serial }
    cond = tirx.And(_core_num() > 0, _core_id() >= 0)
    guarded = tirx.IfThenElse(cond, new_serial, None)

    # SyncN 语句，对应函数末尾的 ``C6678E_SyncN(core_num, core_id);``。
    sync_call = tirx.call_extern("", "C6678E_SyncN", _core_num(), _core_id())
    sync_stmt = tirx.Evaluate(sync_call)

    return tirx.SeqStmt([guarded, sync_stmt])


@prim_func_pass(opt_level=0, name="C6678MulticoreLower")
class C6678MulticoreLower:
    """A.8：把 c6678 host PrimFunc 中的 ``ForKind.PARALLEL`` 循环降级为 BSP 多核派发。

    详细规则见模块 docstring。
    """

    def transform_function(self, func, mod, ctx):  # noqa: D401
        """对单个 PrimFunc 做多核派发改写。

        Parameters
        ----------
        func : tvm.tirx.PrimFunc
            待处理的 PrimFunc。
        mod : tvm.IRModule
            所在模块（本 pass 不使用，仅满足 prim_func_pass 接口）。
        ctx : tvm.transform.PassContext
            Pass 上下文（本 pass 不使用，仅满足 prim_func_pass 接口）。

        Returns
        -------
        tvm.tirx.PrimFunc
            改写后的 PrimFunc。若不属于 c6678、已经 lowered 或 body 中没有
            PARALLEL 循环，则原样返回。
        """
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _already_lowered(func):
            return func
        if not _has_parallel_for(func.body):
            return func

        # 1. 追加 core_mask 形参。
        core_mask_var = tirx.Var(_CORE_MASK_NAME, "int32")
        new_params = list(func.params) + [core_mask_var]

        # 2. 改写 body：找到第一个 ForKind.PARALLEL 节点替换为多核派发结构。
        #    注意：dlight.c6678.Matmul 当前只在最外层 split 出来的 i_outer 上
        #    sch.parallel，不会嵌套；这里仍用 ``replaced`` 标志保证整个 body 只
        #    替换一次，避免未来嵌套场景被重复改写。
        replaced = [False]

        def _post(node):
            if replaced[0]:
                return None
            if isinstance(node, tirx.For) and int(node.kind) == int(tirx.ForKind.PARALLEL):
                replaced[0] = True
                return _build_multicore_body(node, core_mask_var)
            return None

        new_body = tirx.stmt_functor.ir_transform(func.body, None, _post)

        # 3. 重新构造 PrimFunc，保持 ret_type / buffer_map / 其它 attrs 不变。
        new_func = tvm.tirx.PrimFunc(
            params=new_params,
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs,
            span=func.span,
        )
        new_func = new_func.with_attr("c6678.multicore_lowered", True)
        return new_func
