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
"""C6678 硬件参数读取 / 默认值兜底（路线图 §16 中的 A.1）。

本模块是后续所有 c6678 专属 pass / schedule 的"硬件常量唯一事实源"，
设计目标是：

* 单一来源：`from_target(target)` 直接返回 `C6678Config`，避免规则散落；
* 默认值兜底：即便 `Target("c6678")` 没有显式赋值，也能拿到真实硬件参数；
* 纯 Python：不引入新的 C++ 依赖，便于第一阶段快速迭代。

详细设计与上下游关系见 `Test4dsp/learning.md` §16.2 / §16.3 / §16.7。
"""

from __future__ import annotations

from dataclasses import dataclass

from tvm.target import Target

C6678_MAX_CORES = 8

# 默认值与 src/target/target_kind.cc 中 C6678 TargetKind 的 DefaultValue 严格对齐。
# 由于 Target.attrs 在未显式赋值时不会自动填充 DefaultValue，
# 这里再给 Python 侧一份硬编码兜底，确保 pass / schedule 总能拿到完整参数。
_C6678_DEFAULT_ATTRS: dict[str, int] = {
    "core_num": 8,
    "core_freq_mhz": 1250,
    "l1_size": 32 * 1024,
    "l2_size": 1024 * 1024,
    "l2_base_core0": 0x10800000,
    "l2_core_stride": 0x01000000,
    "smc_base": 0x0C000000,
    "smc_size": 0x00800000,
    "ddr_base": 0x80000000,
    "ddr_size": 0x80000000,
    "dma_align_bytes": 64,
    "dma_burst_bytes": 64,
    "dma_max_transfer": 0x7FFFFFFF,
    "vector_bytes": 32,
}


def dtype_nbytes(dtype: str) -> int:
    """Return the byte width of one scalar element for ``dtype``."""
    import tvm

    return tvm.runtime.DataType(dtype).bits // 8


def align_down(value: int, align_bytes: int) -> int:
    """Round ``value`` down to a positive multiple of ``align_bytes``."""
    if align_bytes <= 0:
        raise ValueError(f"align_bytes must be positive, got {align_bytes}")
    return max(align_bytes, (value // align_bytes) * align_bytes)


def l2_dma_block_elems(
    dtype: str,
    l2_size: int,
    num_input_buffers: int,
    num_output_buffers: int,
    reserve_bytes: int = 4096,
    align_bytes: int = 64,
    max_elems: int | None = None,
) -> int:
    """Compute a conservative 1D L2 DMA tile capacity in elements.

    The helper intentionally uses a simple equal-share model.  It is meant for
    first-stage contiguous elementwise/reduction staging, not for the final
    autotuning cost model.
    """
    if num_input_buffers < 0 or num_output_buffers < 0:
        raise ValueError("buffer counts must be non-negative")
    buffer_count = num_input_buffers + num_output_buffers
    if buffer_count <= 0:
        raise ValueError("at least one L2 staging buffer is required")
    elem_bytes = dtype_nbytes(dtype)
    usable_l2 = l2_size - reserve_bytes
    if usable_l2 <= 0:
        raise ValueError(
            f"reserve_bytes={reserve_bytes} leaves no usable L2 from l2_size={l2_size}"
        )
    bytes_per_buffer = align_down(usable_l2 // buffer_count, align_bytes)
    elems = max(1, bytes_per_buffer // elem_bytes)
    if max_elems is not None:
        elems = min(elems, max(1, int(max_elems)))
    return elems


@dataclass(frozen=True)
class C6678Config:
    """C6678 target 上一切下游 pass / schedule 都共享的只读硬件配置。"""

    core_num: int
    core_freq_mhz: int
    l1_size: int
    l2_size: int
    l2_base_core0: int
    l2_core_stride: int
    smc_base: int
    smc_size: int
    ddr_base: int
    ddr_size: int
    dma_align_bytes: int
    dma_burst_bytes: int
    dma_max_transfer: int
    vector_bytes: int

    def l2_address_range(self, core_id: int) -> tuple[int, int]:
        """返回核心 ``core_id`` 的 L2 物理地址区间 ``(base, end_inclusive)``。"""
        if core_id < 0 or core_id >= C6678_MAX_CORES:
            raise ValueError(
                f"core_id 必须落在 [0, {C6678_MAX_CORES})，实际为 {core_id}"
            )
        base = self.l2_base_core0 + core_id * self.l2_core_stride
        end = base + self.l2_size - 1
        return base, end

    def smc_address_range(self) -> tuple[int, int]:
        """返回 SMC 共享区段 ``(base, end_inclusive)``。"""
        return self.smc_base, self.smc_base + self.smc_size - 1

    def ddr_address_range(self) -> tuple[int, int]:
        """返回 DDR 区段 ``(base, end_inclusive)``。"""
        return self.ddr_base, self.ddr_base + self.ddr_size - 1

    def iter_cores_from_mask(self, core_mask: int) -> list[int]:
        """把 ``core_mask`` 位图展开成参与计算的逻辑核心 id 列表。"""
        if core_mask <= 0:
            raise ValueError(f"core_mask 必须为正位图，实际为 {core_mask:#x}")
        if core_mask >> self.core_num:
            raise ValueError(
                f"core_mask {core_mask:#x} 超出核心总数 core{self.core_num - 1}"
            )
        return [i for i in range(self.core_num) if (core_mask >> i) & 1]


def _read_attr(target: Target, name: str) -> int:
    """从 ``target.attrs`` 取值；缺失时回退到默认表。

    由于当前 TVM `Target.attrs` 在未显式赋值时不会自动填入 `DefaultValue`，
    这里必须做一次 Python 侧兜底，否则下游 pass 看到的就是 `KeyError`。
    """
    attrs = target.attrs
    if name in attrs:
        return int(attrs[name])
    if name in _C6678_DEFAULT_ATTRS:
        return _C6678_DEFAULT_ATTRS[name]
    raise KeyError(f"C6678 target 缺失硬件参数 `{name}`，且默认值表未登记")


def from_target(target: str | Target) -> C6678Config:
    """根据 ``target`` 构造 :class:`C6678Config`。

    Parameters
    ----------
    target : str | Target
        必须是 ``c6678`` kind 的 Target，否则抛出 ``ValueError``。

    Returns
    -------
    C6678Config
        本次编译流程中所有 c6678 pass / schedule 共享的硬件配置快照。
    """
    target = Target(target)
    if target.kind.name != "c6678":
        raise ValueError(
            f"C6678Config 仅支持 c6678 target，实际为 {target.kind.name}"
        )
    return C6678Config(
        core_num=_read_attr(target, "core_num"),
        core_freq_mhz=_read_attr(target, "core_freq_mhz"),
        l1_size=_read_attr(target, "l1_size"),
        l2_size=_read_attr(target, "l2_size"),
        l2_base_core0=_read_attr(target, "l2_base_core0"),
        l2_core_stride=_read_attr(target, "l2_core_stride"),
        smc_base=_read_attr(target, "smc_base"),
        smc_size=_read_attr(target, "smc_size"),
        ddr_base=_read_attr(target, "ddr_base"),
        ddr_size=_read_attr(target, "ddr_size"),
        dma_align_bytes=_read_attr(target, "dma_align_bytes"),
        dma_burst_bytes=_read_attr(target, "dma_burst_bytes"),
        dma_max_transfer=_read_attr(target, "dma_max_transfer"),
        vector_bytes=_read_attr(target, "vector_bytes"),
    )
