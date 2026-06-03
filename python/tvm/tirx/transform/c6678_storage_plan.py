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
"""C6678 storage planning pass（路线图 §16 中的 A.2 雏形）。

本 pass 当前为"识别但不改写"形态：

* 仅遍历 `PrimFunc.buffer_map` 中所有外部 buffer，按 `scope` 分类；
* 把分类结果以 `func.attrs["c6678.storage_plan"]` 写回，作为后续
  `C6678DMALegalize` 与 multicore lowering 的输入；
* 不改任何 IR 结构，保证可以独立挂载、单测覆盖，而不影响主 pipeline。

详见 `Test4dsp/learning.md` §16.2 / §16.4 / §16.7。
"""

from __future__ import annotations

import tvm

from .. import c6678_config as _cfg
from .function_pass import prim_func_pass


_LEGAL_DMA_SCOPES = {"l2", "smc", "ddr", "global", ""}


def _normalize_scope(raw_scope: str) -> str:
    """把 buffer scope 统一映射到 ``l2/smc/ddr/global``。

    TVM 中常见的 scope 形态包括 ``""``、``"global"``、``"global.l2"``、
    ``"l2"`` 等；这里抹平差异，便于下游 pass 直接 dict 查表。
    """
    scope = (raw_scope or "global").lower()
    if scope.startswith("global."):
        scope = scope[len("global.") :]
    if scope == "":
        scope = "global"
    return scope


@prim_func_pass(opt_level=0, name="C6678StoragePlan")
class C6678StoragePlan:
    """C6678 专属 storage planning pass（识别阶段）。

    Parameters
    ----------
    strict : bool
        若为 True，遇到非 L2/SMC/DDR/global 的 scope 会直接抛错；
        默认 False，仅记录到结果里供下游决定。
    """

    def __init__(self, strict: bool = False) -> None:
        self.strict = bool(strict)

    def transform_function(self, func, mod, ctx):  # noqa: D401
        del mod, ctx
        target = func.attrs.get("target") if func.attrs is not None else None
        if target is None or target.kind.name != "c6678":
            return func

        cfg = _cfg.from_target(target)
        plan: list[dict[str, int | str]] = []
        for var, buf in func.buffer_map.items():
            scope = _normalize_scope(buf.scope())
            if self.strict and scope not in _LEGAL_DMA_SCOPES:
                raise ValueError(
                    f"buffer `{buf.name}` 的 scope `{scope}` 不在 C6678 允许集合 "
                    f"{sorted(_LEGAL_DMA_SCOPES)} 中"
                )
            entry: dict[str, int | str] = {
                "param": var.name,
                "buffer": buf.name,
                "scope": scope,
            }
            if scope == "smc":
                base, end = cfg.smc_address_range()
                entry["region_base"] = base
                entry["region_end"] = end
            elif scope == "ddr" or scope == "global":
                base, end = cfg.ddr_address_range()
                entry["region_base"] = base
                entry["region_end"] = end
            elif scope == "l2":
                # L2 是 per-core 的，物理地址要等到 multicore lowering
                # 拿到 core_id 后才能确定，这里只标 placeholder。
                entry["region_base"] = -1
                entry["region_end"] = -1
            plan.append(entry)

        return func.with_attr("c6678.storage_plan", tvm.runtime.convert(plan))
