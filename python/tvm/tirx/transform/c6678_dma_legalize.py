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
"""C6678 DMA 合法性校验 pass（路线图 §4.2 A.3）。

设计目标
--------
A.9 ``C6678DMALower`` 把带 ``c6678.dma_load`` 注解的 staging block 改写为
``Evaluate(call_extern("load_row_major_tile", ...))`` 之后，IR 已经具备
"调 BSP DMA 函数搬运 tile"的形态。本 pass 紧跟 ``C6678DMALower`` 之后
挂载，对每条已识别的 DMA extern 调用做**纯静态校验**：

* ``load_row_major_tile``：A.9 给 matmul 等 2D tile 派生出的 DMA 形态，
  10 args（含 ``src_scope`` StringImm）；
* ``dma_trans``：BSP 端 1D 字节连续搬运 wrapper（详见
  ``Test4dsp/examples.md`` 第 1~229 行的 ``void dma_trans(void* src,
  void* dst, int size)``），4 args，主要给 softmax 等沿单一维度切
  ``BLOCK_CAPACITY`` 的算子使用。

校验维度
--------

* **scope 合法性**：``src_scope`` 必须落在 c6678 可见的内存层次内
  （``"global" / "global.l2" / "global.smc"``）；其它 scope 直接报错，
  避免板上 DMA 触发到不存在的地址段。
* **正数性**：``rows / cols / src_ld / elem_size`` 在静态可计算时必须为
  正整数。零或负值表示上游 schedule / lower 计算错误。
* **单次传输上限**：``rows * cols * elem_size`` 不得超过
  ``C6678Config.dma_max_transfer``。一旦超量必须靠未来的 ``dma_trans*``
  族拆分指令；当前阶段直接报错，让上层重新规划 tile。
* **行对齐建议**：``cols * elem_size`` 若静态可计算且非 ``dma_align_bytes``
  的整数倍，发 ``warnings.warn(... C6678DMAAlignmentWarning)``，但不阻断
  编译 —— 这与 BSP 的 DMA 实测兼容性一致（行尾会自动 padding，但效率会降）。

关键设计决策
------------
* **只读校验**：不修改 IR，只决定"通过 / 报错 / 告警"。这样 A.3 在主流水线
  上即便有 bug，最坏情况也只会误报，不会污染 IR。
* **幂等**：用 ``c6678.dma_legalized = True`` 标记，避免重复跑同一个 pass。
* **不依赖 device-id**：完全通过 ``func.attrs["target"]`` 拿
  ``C6678Config``，不去访问运行时 device 信息。

详细位置约束见 ``Test4dsp/learning.md §4.1.1 图 A "A.3 C6678DMALegalize"
节点（紧跟 A.9 C6678DMALower）。
"""

from __future__ import annotations

import warnings

import tvm
from tvm import tirx
from tvm.tirx import stmt_functor

from ..c6678_config import C6678Config, from_target as _config_from_target
from .function_pass import prim_func_pass


# 与 ``C6678DMALower._build_dma_call`` 的 call_extern args[0]（fn_name）对齐。
# 当前接纳两族：
# * ``load_row_major_tile``：A.9 对 2D tile（matmul 输入分块）的搬运，9 args，
#   对齐 BSP 签名 ``void load_row_major_tile(void* src_base, void* dst, int row0,
#   int col0, int rows, int cols, int src_ld, int elem_size)``；
# * ``dma_trans``：BSP 端 1D 字节连续搬运 wrapper（详见
#   ``Test4dsp/examples.md`` 第 1~229 行），4 args，签名为
#   ``void dma_trans(void* src, void* dst, int size)``，源/目的 scope 由
#   schedule 端的 ``c6678.src_scope`` annotation 间接告知，不在 call args 内。
_DMA_CALL_NAMES: frozenset[str] = frozenset(
    {
        "load_row_major_tile",
        "dma_trans",
    }
)

# c6678 codegen 能正确出码的 scope 白名单
_LEGAL_SCOPES: frozenset[str] = frozenset(
    {
        "global",
        "global.l2",
        "global.smc",
    }
)


class C6678DMAAlignmentWarning(UserWarning):
    """``cols * elem_size`` 不是 ``dma_align_bytes`` 整数倍时发出的警告。

    BSP 端 DMA 在行尾会自动 padding，但效率会降；故仅警告不阻断。
    """


def _is_c6678_func(func) -> bool:
    """是否是 ``target.kind.name == "c6678"`` 的 PrimFunc。"""
    if func.attrs is None:
        return False
    target = func.attrs.get("target")
    if target is None:
        return False
    return target.kind.name == "c6678"


def _has_attr_flag(func, key: str) -> bool:
    """检查 ``func.attrs`` 是否有 truthy 的 key 标志（用于幂等）。"""
    if func.attrs is None:
        return False
    flag = func.attrs.get(key)
    if flag is None:
        return False
    return bool(int(flag))


def _try_static_int(expr) -> int | None:
    """把 ``expr`` 静态求值为 ``int``，无法求值时返回 ``None``。

    覆盖三种常见来源：
    * 直接 ``int`` / ``IntImm``；
    * ``tirx.IntImm`` 的常量传播结果；
    * 其它 PrimExpr 直接返回 ``None``，让上层校验跳过该项。
    """
    if isinstance(expr, int):
        return int(expr)
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    return None


def _try_static_str(expr) -> str | None:
    """把 ``expr`` 解析为字符串，仅 ``StringImm`` 与 ``str`` 命中。"""
    if isinstance(expr, str):
        return expr
    if isinstance(expr, tirx.StringImm):
        return str(expr.value)
    return None


def _validate_load_row_major_tile(
    call: tirx.Call, cfg: C6678Config, func_name: str
) -> None:
    """校验单条 ``call_extern("load_row_major_tile", ...)`` 的参数。

    与 ``C6678DMALower._build_dma_call`` 写出的 call 形态严格对齐。源码：

    .. code-block:: python

        tirx.call_extern(
            "",                          # dtype（通过 Call.dtype 暴露，不进 args）
            "load_row_major_tile",       # fn_name
            src_ptr, dst_ptr,            # access_ptr handles
            row0_expr, col0_expr,
            rows_expr, cols_expr,
            src_ld, elem_size,
        )

    经过 ``tvm.tirx.op.call_extern`` 处理后，``Call.args`` 形态为
    ``[fn_name, src_ptr, dst_ptr, row0, col0, rows, cols, src_ld, elem_size]``，
    即 1（fn name）+ 8（params）= **9** 个 args。
    """
    args = list(call.args)
    if len(args) != 9:
        raise ValueError(
            f"[C6678DMALegalize] {func_name}: load_row_major_tile expects 9 call_extern "
            f"args (fn_name + 8 params), got {len(args)}: {args}"
        )

    # args[1..2]: src_ptr / dst_ptr —— 由 access_ptr 生成，不在此 pass 校验
    rows = _try_static_int(args[5])
    cols = _try_static_int(args[6])
    src_ld = _try_static_int(args[7])
    elem_size = _try_static_int(args[8])

    # 1) 正数性（仅在静态时校验，动态 extent 留给运行时）
    for name, value in (
        ("rows", rows),
        ("cols", cols),
        ("src_ld", src_ld),
        ("elem_size", elem_size),
    ):
        if value is None:
            continue
        if value <= 0:
            raise ValueError(
                f"[C6678DMALegalize] {func_name}: {name} must be positive int, "
                f"got {value}"
            )

    # 2) 单次传输字节数上限
    if rows is not None and cols is not None and elem_size is not None:
        bytes_total = rows * cols * elem_size
        if bytes_total > cfg.dma_max_transfer:
            raise ValueError(
                f"[C6678DMALegalize] {func_name}: load_row_major_tile transfer "
                f"size {bytes_total} bytes (rows={rows} cols={cols} "
                f"elem_size={elem_size}) exceeds dma_max_transfer "
                f"{cfg.dma_max_transfer}; consider tile splitting"
            )

    # 3) 行对齐建议（静态可计算时仅警告）
    if cols is not None and elem_size is not None:
        row_bytes = cols * elem_size
        align = cfg.dma_align_bytes
        if align > 0 and row_bytes % align != 0:
            warnings.warn(
                f"[C6678DMALegalize] {func_name}: row_bytes={row_bytes} "
                f"(cols={cols} * elem_size={elem_size}) is not a multiple of "
                f"dma_align_bytes={align}; BSP DMA may auto-pad and degrade",
                C6678DMAAlignmentWarning,
            )


def _validate_dma_trans(
    call: tirx.Call, cfg: C6678Config, func_name: str
) -> None:
    """校验单条 ``call_extern("dma_trans", src, dst, size_bytes)``。

    与 BSP 端 wrapper（``Test4dsp/examples.md`` 第 211~228 行）严格对齐：

    .. code-block:: c

        void dma_trans(void* src, void* dst, int size);

    经过 ``tvm.tirx.op.call_extern`` 处理后，``Call.args`` 形态为
    ``[fn_name, src_ptr, dst_ptr, size_bytes]`` = 1 + 3 = **4** 个 args。

    校验维度（仅 1D，不存在 row alignment 概念）：
    * 单次传输字节数 ``size_bytes`` 静态可计算时必须为正整数；
    * ``size_bytes`` 不得超过 ``cfg.dma_max_transfer``——BSP wrapper 实际会
      自动按 ``0x7fff`` 拆 frame，但 IR 侧仍按硬上限保护，避免 schedule 端
      传入意外的负数 / 0 / 过大值。

    与 2D 路径不同，``dma_trans`` 调用本身**不携带 src_scope**——schedule
    端通过 ``sch.annotate("c6678.src_scope", ...)`` 在 staging block 上独立
    标注，由后续 lowering pass 决定要不要发出额外的 PSC 时钟门控、缓存
    刷写等动作。本 pass 只关心 size 维度的合法性。
    """
    args = list(call.args)
    if len(args) != 4:
        raise ValueError(
            f"[C6678DMALegalize] {func_name}: dma_trans expects 4 call_extern "
            f"args (fn_name + src + dst + size), got {len(args)}: {args}"
        )

    size_bytes = _try_static_int(args[3])
    if size_bytes is not None:
        if size_bytes <= 0:
            raise ValueError(
                f"[C6678DMALegalize] {func_name}: dma_trans size must be positive int, "
                f"got {size_bytes}"
            )
        if size_bytes > cfg.dma_max_transfer:
            raise ValueError(
                f"[C6678DMALegalize] {func_name}: dma_trans transfer size "
                f"{size_bytes} bytes exceeds dma_max_transfer "
                f"{cfg.dma_max_transfer}; consider splitting at schedule level"
            )


def _validate_func_body(body, cfg: C6678Config, func_name: str) -> int:
    """遍历 body，对每条 DMA call_extern 调用 ``_validate_*``，返回命中条数。"""
    hit_count = [0]

    def _visit(node):
        if not isinstance(node, tirx.Call):
            return
        # call_extern 编码：op == "tirx.call_extern"，args[0] 是 fn name StringImm
        op = node.op
        if not hasattr(op, "name") or op.name != "tirx.call_extern":
            return
        if len(node.args) < 1:
            return
        fn_name = _try_static_str(node.args[0])
        if fn_name is None or fn_name not in _DMA_CALL_NAMES:
            return
        hit_count[0] += 1
        if fn_name == "load_row_major_tile":
            _validate_load_row_major_tile(node, cfg, func_name)
        elif fn_name == "dma_trans":
            _validate_dma_trans(node, cfg, func_name)

    stmt_functor.post_order_visit(body, _visit)
    return hit_count[0]


@prim_func_pass(opt_level=0, name="C6678DMALegalize")
class C6678DMALegalize:
    """A.3：c6678 DMA 合法性校验 pass（只读，不改写 IR）。

    挂载位置：紧跟 ``C6678DMALower`` 之后（``_c6678_post_schedule_passes``），
    确保看到的是已经替换为 ``call_extern("load_row_major_tile", ...)`` 的 IR。

    校验失败抛 ``ValueError``；行对齐告警通过 ``C6678DMAAlignmentWarning``
    发出，不阻断编译。
    """

    def transform_function(self, func, mod, ctx):
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _has_attr_flag(func, "c6678.dma_legalized"):
            return func

        target = func.attrs["target"]
        cfg = _config_from_target(target)
        # PrimFunc 没有稳定的 name 属性；尽量从 attrs / global_symbol 取
        func_name = "<c6678_func>"
        if func.attrs is not None:
            gs = func.attrs.get("global_symbol")
            if gs is not None:
                func_name = str(gs)

        _validate_func_body(func.body, cfg, func_name)

        new_func = tvm.tirx.PrimFunc(
            params=list(func.params),
            body=func.body,
            ret_type=func.ret_type,
            buffer_map=func.buffer_map,
            attrs=func.attrs,
            span=func.span,
        )
        return new_func.with_attr("c6678.dma_legalized", True)


__all__ = [
    "C6678DMALegalize",
    "C6678DMAAlignmentWarning",
]
