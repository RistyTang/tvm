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
"""C6678 算子特征抽取（路线图 §4.1.1 图 A "A.4 C6678Features" 节点）。

设计目标
--------
本模块是 ``A.5 dlight.c6678 分发器``（proper 版本）的**输入契约**：把一个
``tirx.PrimFunc`` 翻译成一组对 schedule 选择有意义的、纯结构化、可哈希的
特征（``C6678OpFeatures``）。

关键约束（与 §4.3 选址原则一致）
--------------------------------
* **read-only**：不修改任何 IR 结构，因此放在 ``tirx/analysis/`` 而非
  ``tirx/transform/``；不挂入 ``default_s_tir_pipeline``，A.5 dispatcher
  落地时再串接调用。
* **复用而非重建**：算子分类信息直接复用
  ``tvm.s_tir.dlight.analysis.normalize_prim_func`` 已经计算好的
  ``SBlockInfo``（``dom_kind / read_bufs / write_bufs``），c6678 这一层
  只负责"翻译成 c6678 视角的特征向量 + 给 schedule 模板的提示字段"。
* **target 守卫**：``extract_features`` 在 ``target.kind.name != "c6678"``
  时直接返回 ``None``，避免污染其它 target 的分析路径。

可解锁能力
----------
A.5 dispatcher 落地后，可以基于 ``C6678OpFeatures`` 做：

1. ``op_kind`` × ``dtype`` 派发不同 schedule（如 fp32 gemm 走 L2 staging
   + 内层 _p_ 退化；fp16 gemv 走另一套 cache 策略）。
2. ``tile_hint`` 直接喂给 ``dlight.c6678.Matmul`` 当 default tile，避免
   现在的 ``_pick_factor((32,16,8,4,2,1))`` 硬编码顺序。
3. ``flop_count_static`` / ``static_alloc_l2_bytes`` 让 dispatcher 检查
   一个候选模板是否会撑爆 L2（按 ``c6678_config.l2_size``）。

详细背景见 ``Test4dsp/learning.md`` §4.1.1 图 A、§4.1.4 终态差距表"A.4"行。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from tvm import tirx
from tvm.target import Target

from ..c6678_config import C6678Config, from_target as _config_from_target


# 单一事实源：op_kind 的合法取值；A.5 dispatcher 必须只匹配这里出现过的字符串。
_OP_KIND_GEMM = "gemm"
_OP_KIND_GEMV = "gemv"
_OP_KIND_REDUCTION = "reduction"
_OP_KIND_ELEMENTWISE = "elementwise"
_OP_KIND_INJECTIVE = "injective"
_OP_KIND_UNKNOWN = "unknown"


@dataclass(frozen=True)
class C6678BufferSpec:
    """单个 buffer 的纯结构化描述（哈希友好，便于做 schedule cache key）。

    Attributes
    ----------
    name : str
        Buffer.name（来自 IR；不保证唯一，仅用于诊断）。
    shape : tuple[Optional[int], ...]
        各维 extent；非 IntImm 维填 ``None``。
    dtype : str
        Buffer.dtype 字符串，如 ``"float32"``。
    scope : str
        ``Buffer.scope()``，如 ``"global"`` / ``"global.l2"`` / ``"global.smc"``。
    """

    name: str
    shape: tuple[Optional[int], ...]
    dtype: str
    scope: str


@dataclass(frozen=True)
class C6678OpFeatures:
    """c6678 算子特征向量。

    Attributes
    ----------
    op_kind : str
        算子分类，取值见模块顶部 ``_OP_KIND_*``。
    dom_kind : str
        block 的迭代域 kind，复用 ``SBlockInfo.dom_kind()``，例如 ``"SSR"``。
    dom_extents : tuple[Optional[int], ...]
        各 iter 的静态 extent；非 IntImm 维填 ``None``。
    dtype : str
        主输入 dtype（取 ``block.reads[0].buffer.dtype``，缺读时取 write）。
    is_static_shape : bool
        ``dom_extents`` 是否全部为 IntImm。
    flop_count_static : Optional[int]
        当 ``is_static_shape=True`` 时的静态浮点运算次数估计；否则 None。
        仅按 dom_extents 连乘 × （reduction 算子额外 ×2）粗算，未来可细化。
    read_bufs : tuple[C6678BufferSpec, ...]
        block.reads 的纯结构化描述，按 IR 出现顺序保留。
    write_bufs : tuple[C6678BufferSpec, ...]
        block.writes 的纯结构化描述。
    tile_hint : tuple[Optional[int], ...]
        c6678 视角的默认 tile 提示，长度与 ``dom_extents`` 一致；某维为 None
        表示"该维用整个 extent"。当前实现仅给 ``"SSR"`` 算 (32, 32, 32)，
        其它形态全 None；A.5 dispatcher 后会按 op_kind/dtype 细化。
    static_alloc_l2_bytes : Optional[int]
        若按 ``tile_hint`` 在 L2 暂存"所有读 buffer 一个 tile"的预估字节数；
        非静态 shape 时为 None。dispatcher 用它和 ``C6678Config.l2_size``
        做容量门禁。
    """

    op_kind: str
    dom_kind: str
    dom_extents: tuple[Optional[int], ...]
    dtype: str
    is_static_shape: bool
    flop_count_static: Optional[int]
    read_bufs: tuple[C6678BufferSpec, ...]
    write_bufs: tuple[C6678BufferSpec, ...]
    tile_hint: tuple[Optional[int], ...]
    static_alloc_l2_bytes: Optional[int]


@dataclass(frozen=True)
class C6678PrimFuncFeatures:
    """单个 PrimFunc 的特征汇总。

    Attributes
    ----------
    func_name : str
        PrimFunc 的名字（从 IRModule 拿，单函数路径下用占位符 "main"）。
    blocks : tuple[C6678OpFeatures, ...]
        normalize_prim_func 之后剩下的若干 block 的特征。MVP 阶段我们只关心
        ``len == 1`` 的情形（与 ``dlight.c6678.Matmul.apply`` 保持一致）。
    config : C6678Config
        从 target 解出来的 c6678 硬件配置，dispatcher 选模板时直接用。
    """

    func_name: str
    blocks: tuple[C6678OpFeatures, ...] = field(default_factory=tuple)
    config: C6678Config | None = None


def _is_c6678_target(target: Target) -> bool:
    """守卫：仅 c6678 target 上才返回 True，其它 target 返回 False。"""
    return target is not None and target.kind.name == "c6678"


def _to_static(extent) -> Optional[int]:
    """把单维 extent（IntImm 或 PrimExpr）翻译为 ``int | None``。"""
    if isinstance(extent, int):
        return int(extent)
    if isinstance(extent, tirx.IntImm):
        return int(extent.value)
    return None


def _classify_op(dom_kind: str, read_count: int, write_count: int) -> str:
    """从 ``SBlockInfo.dom_kind()`` 推断 op_kind。

    分类规则（与 §4.1.1 图 A 中 ``A.4 → A.5`` 派发逻辑保持一致）：

    * ``"SSR"`` 且 reads >= 2、writes == 1：``gemm``（M*N*K reduce）。
    * ``"SR"``  且 reads >= 2、writes == 1：``gemv``（M*K reduce 到向量）。
    * 全 ``"S"`` 且 reads == 1、writes == 1：``elementwise``。
    * 全 ``"S"`` 且 reads/writes 任一 != 1：``injective``（多入或多出但仍是
      逐点映射，例如 broadcast / concat 一类）。
    * 含 ``"R"`` 的其它形态：``reduction``。
    * 其它：``unknown``。
    """
    has_r = "R" in dom_kind
    all_s = all(k == "S" for k in dom_kind)
    if dom_kind == "SSR" and read_count >= 2 and write_count == 1:
        return _OP_KIND_GEMM
    if dom_kind == "SR" and read_count >= 2 and write_count == 1:
        return _OP_KIND_GEMV
    if all_s and read_count == 1 and write_count == 1:
        return _OP_KIND_ELEMENTWISE
    if all_s:
        return _OP_KIND_INJECTIVE
    if has_r:
        return _OP_KIND_REDUCTION
    return _OP_KIND_UNKNOWN


def _buffer_spec_from_region(buf_region: tirx.BufferRegion) -> C6678BufferSpec:
    """把 ``BufferRegion.buffer`` 翻译成 ``C6678BufferSpec``。"""
    buf = buf_region.buffer
    shape = tuple(_to_static(s) for s in buf.shape)
    return C6678BufferSpec(
        name=str(buf.name),
        shape=shape,
        dtype=str(buf.dtype),
        scope=buf.scope(),
    )


def _dtype_bytes(dtype: str) -> int:
    """``"float32"`` -> 4；解析失败回退 4，避免 dispatcher 因异常 dtype 崩。"""
    try:
        bits = tirx.IntImm("int32", 0).dtype  # touch to ensure tirx imported
        del bits
        from tvm.runtime import DataType  # 局部 import，避免污染顶层依赖

        return DataType(dtype).bits // 8
    except Exception:  # pylint: disable=broad-except
        return 4


def _compute_tile_hint_and_alloc(
    op_kind: str,
    dom_extents: tuple[Optional[int], ...],
    read_specs: tuple[C6678BufferSpec, ...],
    is_static: bool,
) -> tuple[tuple[Optional[int], ...], Optional[int]]:
    """计算 c6678 视角的默认 tile 与 L2 暂存预估字节数。

    当前实现只覆盖 ``"gemm"`` 一种形态（与 ``dlight.c6678.Matmul.apply``
    保持一致），其它返回全 None。

    gemm 规则：
    * 候选 tile 大小（优先级从高到低）：``(32, 16, 8, 4, 2, 1)``；
    * 三维 (ti, tj, tk) 各自独立挑选**首个能整除对应 dom_extent 的候选**，
      与 ``dlight.c6678.matmul._pick_factor`` 保持一致 —— 这样 features
      给的 tile_hint 直接就是 schedule 会真正使用的 tile，dispatcher 用 1
      代表"该维退化为不切分"；
    * L2 暂存按当前 schedule 最终可能生成的读 buffer staging 全量估算字节数。
      这比单 tile 估算更保守，但和当前 ``cache_read("global.l2")``
      经 ``StorageRewrite`` 后的实际 alloc 口径一致，避免 L2 gate 低估。
    """
    n = len(dom_extents)
    if op_kind != _OP_KIND_GEMM or n != 3 or not is_static:
        return (tuple([None] * n), None)

    m, k_or_n, k = dom_extents  # SSR: (M, N, K)
    # mypy/pylint 不知道 is_static => 各维都是 int，这里再做一次保护
    if m is None or k_or_n is None or k is None:
        return (tuple([None] * n), None)

    candidates = (32, 16, 8, 4, 2, 1)

    def _pick_divisible(extent: int) -> int:
        for f in candidates:
            if f > 0 and extent % f == 0:
                return f
        return 1

    ti = _pick_divisible(int(m))
    tj = _pick_divisible(int(k_or_n))
    tk = _pick_divisible(int(k))

    if not read_specs:
        return ((ti, tj, tk), None)

    total_bytes = 0
    for spec in read_specs:
        if not spec.shape or any(dim is None for dim in spec.shape):
            return ((ti, tj, tk), None)
        elem_count = 1
        for dim in spec.shape:
            elem_count *= int(dim)
        total_bytes += elem_count * _dtype_bytes(spec.dtype)

    return ((ti, tj, tk), total_bytes)


def _features_for_block(sch_block_info, sch) -> C6678OpFeatures:
    """把单个 ``SBlockInfo`` 翻译为 ``C6678OpFeatures``。"""
    dom_kind = sch_block_info.dom_kind()
    dom_extents = tuple(_to_static(d) for d in sch_block_info.dom())
    is_static = all(d is not None for d in dom_extents)

    block_stmt = sch.get(sch_block_info.block_rv)
    read_specs = tuple(_buffer_spec_from_region(r) for r in block_stmt.reads)
    write_specs = tuple(_buffer_spec_from_region(w) for w in block_stmt.writes)

    if read_specs:
        dtype = read_specs[0].dtype
    elif write_specs:
        dtype = write_specs[0].dtype
    else:
        dtype = "unknown"

    op_kind = _classify_op(dom_kind, len(read_specs), len(write_specs))

    flop_count: Optional[int] = None
    if is_static:
        prod = 1
        for d in dom_extents:
            prod *= int(d)  # type: ignore[arg-type]
        # reduction（含 GEMM/GEMV）一次累加 = 1 mul + 1 add，按 2 FLOP 计
        flop_count = prod * 2 if "R" in dom_kind else prod

    tile_hint, alloc_bytes = _compute_tile_hint_and_alloc(
        op_kind=op_kind,
        dom_extents=dom_extents,
        read_specs=read_specs,
        is_static=is_static,
    )

    return C6678OpFeatures(
        op_kind=op_kind,
        dom_kind=dom_kind,
        dom_extents=dom_extents,
        dtype=dtype,
        is_static_shape=is_static,
        flop_count_static=flop_count,
        read_bufs=read_specs,
        write_bufs=write_specs,
        tile_hint=tile_hint,
        static_alloc_l2_bytes=alloc_bytes,
    )


def extract_features(
    func: tirx.PrimFunc,
    target: Target,
    func_name: str = "main",
) -> Optional[C6678PrimFuncFeatures]:
    """从单个 PrimFunc 抽 c6678 算子特征。

    Parameters
    ----------
    func : tirx.PrimFunc
        待分析的 IR；要求是 ``s_tir`` 风格（仍含 SBlock）的 PrimFunc。
        典型时机：``ApplyDefaultSchedule`` 之前调用本函数获取特征，再据此
        选 schedule rule。
    target : Target
        目标硬件描述。仅 ``target.kind.name == "c6678"`` 才会真正抽特征。
    func_name : str
        诊断用，单函数路径下默认 ``"main"``。

    Returns
    -------
    features : C6678PrimFuncFeatures | None
        非 c6678 target、或 ``normalize_prim_func`` 失败时返回 ``None``，
        让 dispatcher 直接跳过 c6678 专属 schedule。
    """
    if not _is_c6678_target(target):
        return None
    if not isinstance(func, tirx.PrimFunc):
        return None

    # 局部 import：避免 tirx 包顶层就拉起整个 s_tir.dlight 子树。
    from tvm import s_tir
    from tvm.s_tir.dlight.analysis import normalize_prim_func

    sch = s_tir.Schedule(func)
    block_infos = normalize_prim_func(sch)
    if block_infos is None:
        return None

    blocks = tuple(_features_for_block(bi, sch) for bi in block_infos)
    cfg = _config_from_target(target)
    return C6678PrimFuncFeatures(func_name=func_name, blocks=blocks, config=cfg)


__all__ = [
    "C6678BufferSpec",
    "C6678OpFeatures",
    "C6678PrimFuncFeatures",
    "extract_features",
]
