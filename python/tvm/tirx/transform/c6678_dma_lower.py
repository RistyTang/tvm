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
"""C6678 DMA 降级 pass（路线图 §4.8 A.9）。

为什么需要这两个 pass：``dlight.c6678.Matmul`` schedule 阶段（A.10 step1）
用 ``cache_read("global.l2") + compute_at(k_outer) + sch.annotate(...)``
注入 staging block 和 ``c6678.dma_load = "load_row_major_tile"`` 注解；
然而 c6678 是 ``kDLCPU`` 但它没有运行时 device-id 信息，``LowerTVMBuiltin``
在遇到 ``scope != "global"`` 的 ``AllocBuffer`` 时会触发
``Unknown device id in current IR``。

A.9 拆成两段：

1. ``C6678DMALower``：紧贴 ``ConvertBlocksToOpaque`` **之前**执行，要求
   ``SBlockRealize.iter_values`` 仍然存在；把 ``for ax0, ax1 in T.grid(...) :
   SBlockRealize(staging_block)`` 整体替换为
   ``Evaluate(call_extern("load_row_major_tile", src.data, dst.data,
   row0, col0, rows, cols, src_ld, sizeof_elem, src_scope))``。
2. ``C6678AnnotateL2Alloc``：在 ``LowerOpaqueBlock`` **之后**、
   ``LowerTVMBuiltin`` **之前**执行；此时 ``SBlock alloc_buffers`` 已被
   降级为顶层 ``AllocBuffer`` Stmt。本 pass 给 ``buffer.scope() == "global.l2"``
   的 ``AllocBuffer`` 追加注解 ``disable_lower_builtin = True``，让
   ``LowerTVMBuiltin`` 直接保留这条 alloc，不去查 device-id（见
   ``src/tirx/transform/lower_tvm_builtin.cc::VisitStmt_(const AllocBufferNode*)``
   第 244-248 行）。codegen_c6678 端会把它当成普通 stack alloc 输出。

详细 spec 见 ``Test4dsp/learning.md §4.8.3``。
"""

from __future__ import annotations

from dataclasses import dataclass

import tvm
from tvm import tirx
from tvm.tirx import stmt_functor

from ..c6678_config import from_target as _config_from_target
from .function_pass import prim_func_pass


_DMA_LOAD_KEY = "c6678.dma_load"
_DMA_SRC_BUFFER_KEY = "c6678.src_buffer"
_DMA_SRC_SCOPE_KEY = "c6678.src_scope"
_DMA_LOAD_VALUE = "load_row_major_tile"
# PR-S1：1D 字节连续搬运形态，配合 BSP wrapper
# ``void dma_trans(void* src, void* dst, int size)``。schedule 端用
# ``sch.annotate(staging_blk, "c6678.dma_load", "dma_trans")`` 触发该路径。
_DMA_TRANS_VALUE = "dma_trans"
_L2_SCOPE = "global.l2"
_DISABLE_LOWER_BUILTIN_KEY = "disable_lower_builtin"
_L2_STATIC_ALLOC_KEY = "c6678.l2_static_alloc"
_L2_BASE_CORE0_KEY = "c6678.l2_base_core0"
_L2_CORE_STRIDE_KEY = "c6678.l2_core_stride"


@dataclass
class _DMACompactInfo:
    """Tile-local L2 compact metadata for one 1D ``dma_trans`` staging buffer."""

    old_buf: tirx.Buffer
    compact_buf: tirx.Buffer
    start_expr: tirx.PrimExpr
    tile_extent: tirx.PrimExpr
    valid_extent: tirx.PrimExpr


def _is_c6678_func(func):
    """是否是 target.kind.name == "c6678" 的 PrimFunc。"""
    if func.attrs is None:
        return False
    target = func.attrs.get("target")
    if target is None:
        return False
    return target.kind.name == "c6678"


def _has_attr_flag(func, key):
    """检查 func.attrs 是否有 truthy 的 key 标志（用于幂等）。"""
    if func.attrs is None:
        return False
    flag = func.attrs.get(key)
    if flag is None:
        return False
    return bool(int(flag))


def _has_dma_annotation(stmt):
    """body 中是否存在带 c6678.dma_load 注解的 SBlock（``load_row_major_tile``
    或 ``dma_trans``）。"""
    found = [False]

    def _visit(node):
        if isinstance(node, tirx.SBlock):
            anns = node.annotations
            if anns is not None and _DMA_LOAD_KEY in anns:
                found[0] = True

    stmt_functor.post_order_visit(stmt, _visit)
    return found[0]


def _zero_substitute(expr, vars_to_zero):
    """把 expr 中出现的 vars_to_zero 各替换为 0。"""
    if not vars_to_zero:
        return expr
    vmap = {v: tirx.IntImm(v.dtype, 0) for v in vars_to_zero}
    return stmt_functor.substitute(expr, vmap)


def _buffer_type_name(buf):
    """Return the decl_buffer ``buffer_type`` name for ``buf``."""
    return "auto_broadcast" if int(buf.buffer_type) == 2 else ""


def _make_compact_buffer(old_buf, compact_extent):
    """Create a tile-local view for a 1D L2 staging buffer.

    The new buffer keeps the original storage scope and dtype, but its logical
    shape is reduced from the full problem size to the DMA tile extent.  A new
    data Var is used so downstream StorageRewrite can still merge compact L2
    buffers and synthesize the proper offset via access_ptr.
    """
    from tvm.ir import PointerType

    old_ptr_type = old_buf.data.type_annotation
    storage_scope = old_buf.scope()
    new_data = tirx.Var(
        old_buf.data.name,
        PointerType(old_ptr_type.element_type, storage_scope),
    )
    return tirx.decl_buffer(
        shape=(compact_extent,),
        dtype=old_buf.dtype,
        name=old_buf.name,
        data=new_data,
        strides=None,
        elem_offset=old_buf.elem_offset,
        scope=storage_scope,
        data_alignment=old_buf.data_alignment,
        offset_factor=old_buf.offset_factor,
        buffer_type=_buffer_type_name(old_buf),
    )


def _build_dma_call(staging_realize, outer_for_ax0, outer_for_ax1):
    """把 ``for ax0, ax1: SBlockRealize`` 替换为 Evaluate(call_extern(...))。

    - row0/col0：把 staging block 的 ``iter_values`` 中所有 ``ax0``/``ax1`` 变量
      替换为 0 后得到。例如 staging block ``A_global.l2`` 的 iter_values
      为 ``[i_0*32 + ax0, k_0*32 + ax1]``，零代入后 row0=``i_0*32``、
      col0=``k_0*32``。
    - rows/cols：来自外层 ``For(ax0)``/``For(ax1)`` 的 ``extent``。
    - src_ld：源 buffer 的最后一维 shape（cache_read 不会改写源 layout）。
    - elem_size：源 buffer 的 dtype 字节数。
    - src_scope：从 schedule 阶段注入的 ``c6678.src_scope`` 注解读取，缺省 "global"。
    """
    block = staging_realize.block
    iter_values = list(staging_realize.iter_values)
    assert len(iter_values) == 2, (
        "load_row_major_tile expects 2D staging block, got "
        + str(iter_values) + " for " + block.name_hint
    )

    ax_vars = [outer_for_ax0.loop_var, outer_for_ax1.loop_var]
    row0_expr = _zero_substitute(iter_values[0], ax_vars)
    col0_expr = _zero_substitute(iter_values[1], ax_vars)

    rows_expr = outer_for_ax0.extent
    cols_expr = outer_for_ax1.extent

    src_buf = block.reads[0].buffer
    dst_buf = block.writes[0].buffer
    src_ld = src_buf.shape[-1]

    anns = dict(block.annotations) if block.annotations else {}
    elem_bytes = tvm.runtime.DataType(src_buf.dtype).bits // 8
    elem_size = tirx.IntImm("int32", elem_bytes)

    # 关键设计：用 access_ptr 而非裸 .data。
    # - 裸 dst_buf.data 是不透明 Var，StorageRewrite 看不见数据流，
    #   合并 A_global.l2 + B_global.l2 时不会自动加 elem_offset，
    #   导致 B 段的 DMA 第二参数误指向 A 段头部（详见 learning.md §4.1.4
    #   A.9 follow-up + §4.8.3 第 3 条 follow-up 段）。
    # - access_ptr("r"/"w") 走 builtin tvm_access_ptr，
    #   storage_rewrite.cc:1601-1625 专门会合成正确的偏移参数；
    #   随后 LowerIntrin 把 tvm_access_ptr 降级为 address_of(BufferLoad)，
    #   C codegen 出 `&A_global_l2[16384]` 形态。
    src_ptr = src_buf.access_ptr("r")
    dst_ptr = dst_buf.access_ptr("w")

    call = tirx.call_extern(
        "",
        "load_row_major_tile",
        src_ptr,
        dst_ptr,
        row0_expr,
        col0_expr,
        rows_expr,
        cols_expr,
        src_ld,
        elem_size,
    )
    return tirx.Evaluate(call)


def _build_dma_trans_call(staging_realize, outer_for_ax0):
    """把 ``for ax0: SBlockRealize(staging)`` 替换为
    ``Evaluate(call_extern("dma_trans", src_ptr, dst_ptr, size_bytes))``。

    与 ``_build_dma_call`` 的关键差异：

    - **1D 连续视角**：``dma_trans`` BSP wrapper 不关心多维 layout，只接受
      ``(src, dst, size_bytes)``。按用户 PR-S1 设计约定（参见
      ``user_read.md §5.1``），dma_trans **仅用于** softmax 等沿 axis 切
      BLOCK_CAPACITY 的 1 层 ``for ax: SBlockRealize`` 形态；2D tile 形态
      继续走 ``load_row_major_tile``。
    - **不写 src_scope**：BSP wrapper 内部用固定 DMA0 + ``DNUM`` 自管时钟
      / 队列；schedule 端的 ``c6678.src_scope`` 仅留给 A.3 兜底校验，不进
      call args。
    - **同样用 access_ptr** 让 ``StorageRewrite`` 自动合成 elem_offset。

    Parameters
    ----------
    staging_realize : tirx.SBlockRealize
        ``c6678.dma_load == "dma_trans"`` 注解所在 staging block 的实现节点。
    outer_for_ax0 : tirx.For
        包裹该 staging 的唯一一层 ``For``，``extent`` 即 staging block 的
        总元素数。
    """
    block = staging_realize.block
    iter_values = list(staging_realize.iter_values)
    assert len(iter_values) == 1, (
        "dma_trans expects 1D staging block, got "
        + str(iter_values) + " for " + block.name_hint
    )

    src_buf = block.reads[0].buffer
    dst_buf = block.writes[0].buffer
    elem_bytes = tvm.runtime.DataType(src_buf.dtype).bits // 8

    start_expr = _zero_substitute(iter_values[0], [outer_for_ax0.loop_var])
    valid_extent = outer_for_ax0.extent
    if len(src_buf.shape) == 1:
        valid_extent = tirx.Min(valid_extent, src_buf.shape[0] - start_expr)
    compact_dst_buf = _make_compact_buffer(dst_buf, outer_for_ax0.extent)

    # size_bytes = valid_extent * elem_bytes.  For tail tiles this avoids
    # copying beyond the logical buffer even when the split factor is larger
    # than the remaining element count.
    size_expr = tirx.IntImm("int32", elem_bytes) * valid_extent

    src_ptr = src_buf.access_ptr("r", offset=start_expr, extent=valid_extent)
    dst_ptr = compact_dst_buf.access_ptr(
        "w",
        offset=tirx.IntImm("int32", 0),
        extent=valid_extent,
    )

    call = tirx.call_extern(
        "",
        "dma_trans",
        src_ptr,
        dst_ptr,
        size_expr,
    )
    compact_info = _DMACompactInfo(
        old_buf=dst_buf,
        compact_buf=compact_dst_buf,
        start_expr=start_expr,
        tile_extent=outer_for_ax0.extent,
        valid_extent=valid_extent,
    )
    return tirx.Evaluate(call), compact_info


def _same_prim_expr(lhs, rhs):
    """Best-effort structural equality for PrimExpr-like nodes."""
    try:
        return tvm.ir.structural_equal(lhs, rhs)
    except (TypeError, ValueError):
        return str(lhs) == str(rhs)


def _add_compact_info(compact_infos, info):
    """Record compact info and reject inconsistent remaps for one buffer."""
    old = compact_infos.get(info.old_buf.data)
    if old is not None:
        if not _same_prim_expr(old.tile_extent, info.tile_extent):
            raise ValueError(
                "C6678 DMA compact found inconsistent tile extents for buffer "
                f"{info.old_buf.name}: {old.tile_extent} vs {info.tile_extent}"
            )
        return
    compact_infos[info.old_buf.data] = info


def _peel_outer_fors(node):
    """从 ``node`` 起向内剥若干层 ``For``，返回 ``(for_list, innermost)``。

    用于支持不同 staging 形态的统一识别：load_row_major_tile 期望剥到 2 层
    （ax0/ax1），dma_trans 期望剥到 1 层（ax）。具体匹配的层数由调用方在
    ``_rewrite_dma_body`` 内根据 annotation value 校验。
    """
    fors = []
    while isinstance(node, tirx.For):
        fors.append(node)
        node = node.body
    return fors, node


def _rewrite_dma_body(body):
    """对 body 做 DMA 注解 → call_extern 的整体改写。

    支持两种 staging 形态：

    - ``c6678.dma_load == "load_row_major_tile"``：要求**严格 2 层 For** 包
      ``SBlockRealize``，落到 ``_build_dma_call``；
    - ``c6678.dma_load == "dma_trans"``：要求**严格 1 层 For** 包
      ``SBlockRealize``（softmax 沿 axis 切 BLOCK_CAPACITY 形态），落到
      ``_build_dma_trans_call``。
    """

    compact_infos = {}

    def postorder(node):
        if not isinstance(node, tirx.For):
            return None

        outer_fors, innermost = _peel_outer_fors(node)
        if not isinstance(innermost, tirx.SBlockRealize):
            return None
        block = innermost.block
        anns = block.annotations
        if anns is None or _DMA_LOAD_KEY not in anns:
            return None

        ann_val = str(anns[_DMA_LOAD_KEY])
        if ann_val == _DMA_LOAD_VALUE:
            # 严格 2 层 For，与 _build_dma_call 的要求保持一致
            if len(outer_fors) != 2:
                return None
            return _build_dma_call(innermost, outer_fors[0], outer_fors[1])
        if ann_val == _DMA_TRANS_VALUE:
            # 严格 1 层 For，对齐 user_read.md §5.1 约定
            if len(outer_fors) != 1:
                return None
            new_stmt, compact_info = _build_dma_trans_call(innermost, outer_fors[0])
            _add_compact_info(compact_infos, compact_info)
            return new_stmt
        return None

    new_body = stmt_functor.ir_transform(body, None, postorder)
    return new_body, compact_infos


def _rewrite_buffer_region(region, buf_remap_by_data, extent_remap_by_data):
    """Rewrite BufferRegion metadata when its buffer is compacted."""
    new_buf = buf_remap_by_data.get(region.buffer.data)
    if new_buf is None:
        return region
    new_extent = extent_remap_by_data[region.buffer.data]
    return tirx.BufferRegion(
        new_buf,
        [tvm.ir.Range.from_min_extent(tirx.IntImm("int32", 0), new_extent)],
    )


def _apply_dma_l2_compact(body, compact_infos):
    """Apply tile-local L2 remap for 1D ``dma_trans`` staging buffers.

    This is deliberately narrower than TVM's general CompactBufferAllocation:
    it only rewrites buffers produced by c6678 1D DMA staging.  The pass runs
    before ``ConvertBlocksToOpaque`` so ``start_expr`` is still available.
    """
    if not compact_infos:
        return body

    buf_remap_by_data = {
        info.old_buf.data: info.compact_buf for info in compact_infos.values()
    }
    start_remap_by_data = {
        info.old_buf.data: info.start_expr for info in compact_infos.values()
    }
    extent_remap_by_data = {
        info.old_buf.data: info.tile_extent for info in compact_infos.values()
    }
    var_remap = {
        info.old_buf.data: info.compact_buf.data for info in compact_infos.values()
    }

    def _local_index(buf, indices):
        if len(indices) != 1:
            raise ValueError(
                "C6678 DMA compact currently supports only 1D L2 buffers, got "
                f"{len(indices)}D access for {buf.name}"
            )
        return indices[0] - start_remap_by_data[buf.data]

    def postorder(node):
        if isinstance(node, tirx.SBlock):
            alloc_buffers = [
                buf_remap_by_data.get(buf.data, buf) for buf in node.alloc_buffers
            ]
            reads = [
                _rewrite_buffer_region(region, buf_remap_by_data, extent_remap_by_data)
                for region in node.reads
            ]
            writes = [
                _rewrite_buffer_region(region, buf_remap_by_data, extent_remap_by_data)
                for region in node.writes
            ]
            changed = (
                any(new is not old for new, old in zip(alloc_buffers, node.alloc_buffers))
                or any(new is not old for new, old in zip(reads, node.reads))
                or any(new is not old for new, old in zip(writes, node.writes))
            )
            if not changed:
                return None
            return tirx.SBlock(
                list(node.iter_vars),
                reads,
                writes,
                node.name_hint,
                node.body,
                node.init,
                alloc_buffers,
                list(node.match_buffers),
                dict(node.annotations),
                getattr(node, "span", None),
            )
        if isinstance(node, tirx.AllocBuffer):
            new_buf = buf_remap_by_data.get(node.buffer.data)
            if new_buf is None:
                return None
            return tirx.AllocBuffer(new_buf, node.annotations)
        if isinstance(node, tirx.BufferLoad):
            new_buf = buf_remap_by_data.get(node.buffer.data)
            if new_buf is None:
                return None
            local_idx = _local_index(node.buffer, list(node.indices))
            return tirx.BufferLoad(new_buf, [local_idx], node.predicate)
        if isinstance(node, tirx.BufferStore):
            new_buf = buf_remap_by_data.get(node.buffer.data)
            if new_buf is None:
                return None
            local_idx = _local_index(node.buffer, list(node.indices))
            return tirx.BufferStore(new_buf, node.value, [local_idx], node.predicate)
        return None

    new_body = stmt_functor.ir_transform(body, None, postorder)
    if var_remap:
        new_body = stmt_functor.substitute(new_body, var_remap)
    return new_body


def _annotate_l2_alloc(body, cfg):
    """把 buffer.scope() == "global.l2" 的 AllocBuffer 改写成 scope='global'。

    背景见模块 docstring + ``learning.md §4.8.3``：
    - ``LowerTVMBuiltin`` 在 ``scope != "global"`` 时会查 device-id，c6678
      是 kDLCPU 但 IR 里没填 device id，触发 ``Unknown device id in current IR``。
    - 即便给 alloc 加 ``disable_lower_builtin = True`` 注解让 LowerTVMBuiltin
      原样保留，``CodeGenC::PrintStorageScope`` 仍有 ``ICHECK_EQ(scope, "global")``。
    - 所以必须在 codegen 之前真的把 ``Buffer.data`` 的 ``PointerType.storage_scope``
      改成 ``"global"``，并把 body 中所有 ``BufferLoad/BufferStore/AllocBuffer``
      指向旧 Buffer 对象的引用一并替换为新 Buffer；同时给 alloc 留下
      ``disable_lower_builtin = True`` 注解作为 LowerTVMBuiltin 端的兜底。
    """
    import os
    from tvm.ir import PointerType
    _dbg = os.environ.get("C6678_DMA_DEBUG", "")
    seen = [0, 0, 0, 0]  # AllocBuffer total / global.l2 hits / BufferLoad rewrites / BufferStore rewrites
    var_remap = {}
    buf_remap_by_data = {}

    def _rebuild_buffer_global(old_buf):
        old_data = old_buf.data
        old_ptr_type = old_data.type_annotation
        new_ptr_type = PointerType(old_ptr_type.element_type, "global")
        new_data = tirx.Var(old_data.name, new_ptr_type)
        new_buf = tirx.decl_buffer(
            shape=old_buf.shape,
            dtype=old_buf.dtype,
            name=old_buf.name,
            data=new_data,
            strides=old_buf.strides if old_buf.strides else None,
            elem_offset=old_buf.elem_offset,
            scope="global",
            data_alignment=old_buf.data_alignment,
            offset_factor=old_buf.offset_factor,
            buffer_type="auto_broadcast" if int(old_buf.buffer_type) == 2 else "",
        )
        return new_data, new_buf

    def alloc_postorder(node):
        if not isinstance(node, tirx.AllocBuffer):
            return None
        seen[0] += 1
        old_buf = node.buffer
        if old_buf.scope() != _L2_SCOPE:
            return None
        seen[1] += 1
        new_data, new_buf = _rebuild_buffer_global(old_buf)
        var_remap[old_buf.data] = new_data
        buf_remap_by_data[old_buf.data] = new_buf
        old_anns = {}
        try:
            if node.annotations is not None:
                old_anns = dict(node.annotations)
        except AttributeError:
            old_anns = {}
        old_anns[_DISABLE_LOWER_BUILTIN_KEY] = tvm.runtime.convert(True)
        old_anns[_L2_STATIC_ALLOC_KEY] = tvm.runtime.convert(True)
        old_anns[_L2_BASE_CORE0_KEY] = tirx.IntImm("int32", cfg.l2_base_core0)
        old_anns[_L2_CORE_STRIDE_KEY] = tirx.IntImm("int32", cfg.l2_core_stride)
        return tirx.AllocBuffer(new_buf, old_anns)

    new_body = stmt_functor.ir_transform(body, None, alloc_postorder)
    if not buf_remap_by_data:
        if _dbg:
            print("[A.9 step2] AllocBuffer total=", seen[0], " global.l2=", seen[1])
        return new_body

    def buf_postorder(node):
        if isinstance(node, tirx.BufferLoad):
            new_buf = buf_remap_by_data.get(node.buffer.data)
            if new_buf is None:
                return None
            seen[2] += 1
            return tirx.BufferLoad(new_buf, list(node.indices), node.predicate)
        if isinstance(node, tirx.BufferStore):
            new_buf = buf_remap_by_data.get(node.buffer.data)
            if new_buf is None:
                return None
            seen[3] += 1
            return tirx.BufferStore(
                new_buf, node.value, list(node.indices), node.predicate
            )
        return None

    new_body = stmt_functor.ir_transform(new_body, None, buf_postorder)
    if var_remap:
        new_body = stmt_functor.substitute(new_body, var_remap)

    if _dbg:
        print(
            "[A.9 step2] AllocBuffer total=", seen[0],
            " global.l2=", seen[1],
            " BufferLoad rewrites=", seen[2],
            " BufferStore rewrites=", seen[3],
            " var_remap=", len(var_remap),
        )
    return new_body


@prim_func_pass(opt_level=0, name="C6678DMALower")
class C6678DMALower:
    """A.9 step1：把带 c6678.dma_load 注解的 staging block 替换为 call_extern。"""

    def transform_function(self, func, mod, ctx):
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _has_attr_flag(func, "c6678.dma_lowered"):
            return func
        if not _has_dma_annotation(func.body):
            return func

        new_body, compact_infos = _rewrite_dma_body(func.body)
        new_body = _apply_dma_l2_compact(new_body, compact_infos)
        if _has_dma_annotation(new_body):
            raise ValueError(
                "C6678DMALower failed to lower all c6678.dma_load annotations; "
                "check that each annotated staging block matches the expected "
                "load_row_major_tile(2D) or dma_trans(1D) loop shape."
            )
        new_func = tvm.tirx.PrimFunc(
            params=list(func.params),
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs,
            span=func.span,
        )
        return new_func.with_attr("c6678.dma_lowered", True)


@prim_func_pass(opt_level=0, name="C6678AnnotateL2Alloc")
class C6678AnnotateL2Alloc:
    """A.9 step2：给 scope='global.l2' 的 AllocBuffer 追加 disable_lower_builtin=True。"""

    def transform_function(self, func, mod, ctx):
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _has_attr_flag(func, "c6678.l2_alloc_annotated"):
            return func

        cfg = _config_from_target(func.attrs["target"])
        new_body = _annotate_l2_alloc(func.body, cfg)
        new_func = tvm.tirx.PrimFunc(
            params=list(func.params),
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs,
            span=func.span,
        )
        return new_func.with_attr("c6678.l2_alloc_annotated", True)


def _annotate_global_alloc(body):
    """给 scope='global' 的 AllocBuffer 追加 disable_lower_builtin=True。

    背景（PR-S2 phase A）：除 matmul 这种"无中间 buffer"的极简 demo 外，多数
    c6678 PrimFunc 在 schedule 后会在 root SBlock 留下 ``alloc_buffers``
    （softmax 的 ``T_max / T_exp / T_sum``、layernorm 的均值方差缓存等），
    它们经 ``PlanAndUpdateBufferAllocationLocation + LowerOpaqueBlock`` 后
    成为顶层 ``AllocBuffer(scope='global')``。

    ``LowerTVMBuiltin`` 在 ``scope == "global"`` 且 alloc 总字节数小于
    ``runtime::kMaxStackAlloca``（1024）时会原样保留为 stack alloca；超出
    阈值会触发 ``TVMBackendAllocWorkspace`` 路径，要求 IR 携带 ``device_id``
    AttrStmt，但 c6678 是 bare-metal kDLCPU 端没有这条信息，于是抛
    ``Unknown device id in current IR``。

    既然 c6678 codegen（``CodeGenC6678 / CodeGenC::VisitStmt_(AllocBufferNode)``）
    本来就把 ``AllocBuffer`` 直接吐成 ``float buf[N];`` 形态、不依赖 workspace
    runtime，那就给所有 c6678 的 ``scope='global'`` AllocBuffer 加上
    ``disable_lower_builtin = True``：让 ``LowerTVMBuiltin`` 直接 ``return stmt;``
    跳过 workspace 改写（详见
    ``src/tirx/transform/lower_tvm_builtin.cc::VisitStmt_(AllocBufferNode)``
    第 244-248 行的 early-return 路径）。

    注意：本 pass 只追加 annotation，**不**改写 ``Buffer.data`` 的
    ``PointerType.storage_scope`` —— 因为已经是 ``global``，下游 codegen 不会
    在 ``PrintStorageScope`` 处 ICHECK 失败。需要 ``global.l2`` rebuild 的
    场景由 ``C6678AnnotateL2Alloc`` 已经处理过，本 pass 跑在它之后。
    """

    def alloc_postorder(node):
        if not isinstance(node, tirx.AllocBuffer):
            return None
        scope = node.buffer.scope()
        if scope != "global":
            return None
        old_anns = {}
        try:
            if node.annotations is not None:
                old_anns = dict(node.annotations)
        except AttributeError:
            old_anns = {}
        if _DISABLE_LOWER_BUILTIN_KEY in old_anns:
            return None
        old_anns[_DISABLE_LOWER_BUILTIN_KEY] = tvm.runtime.convert(True)
        return tirx.AllocBuffer(node.buffer, old_anns)

    return stmt_functor.ir_transform(body, None, alloc_postorder)


@prim_func_pass(opt_level=0, name="C6678AnnotateGlobalAlloc")
class C6678AnnotateGlobalAlloc:
    """PR-S2 phase A：给所有 c6678 PrimFunc 的 scope='global' AllocBuffer
    追加 ``disable_lower_builtin = True``，避免 ``LowerTVMBuiltin`` 在大块
    中间 buffer（如 softmax 的 ``T_exp``）上触发 ``Unknown device id`` 报错。

    设计要点：
    - 仅对 ``target.kind == c6678`` 的 PrimFunc 生效，不影响其它 backend；
    - 跑在 ``C6678AnnotateL2Alloc`` 之后、``LowerTVMBuiltin`` 之前；
    - 幂等：用 ``c6678.global_alloc_annotated`` 标记。
    """

    def transform_function(self, func, mod, ctx):
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _has_attr_flag(func, "c6678.global_alloc_annotated"):
            return func

        new_body = _annotate_global_alloc(func.body)
        new_func = tvm.tirx.PrimFunc(
            params=list(func.params),
            body=new_body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs,
            span=func.span,
        )
        return new_func.with_attr("c6678.global_alloc_annotated", True)
