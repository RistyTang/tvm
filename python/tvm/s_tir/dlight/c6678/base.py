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
"""Base schedule rule for C6678 DSP target.

参考 `s_tir/dlight/cpu/base.py`，唯一区别是把 `target.kind.name == "llvm"`
换成 `"c6678"`。后续如果需要在所有 c6678 规则之间共享工具函数（例如对齐到
`vector_bytes`、按 `core_num` 切外层循环），都集中放到这里以避免散落。
"""

from tvm.target import Target

from ..base import ScheduleRule


class C6678ScheduleRule(ScheduleRule):  # pylint: disable=too-few-public-methods
    """限定只在 C6678 DSP target 上生效的 ScheduleRule 基类。"""

    def is_target_available(self, target: Target) -> bool:
        """检查 target 是否为 c6678 DSP，并叠加上游基类的可用性判断。

        Parameters
        ----------
        target : Target
            当前调度规则所面向的编译目标。

        Returns
        -------
        available : bool
            仅当上游基类认为可用且 ``target.kind.name == "c6678"`` 时返回 True。
        """
        return super().is_target_available(target) and target.kind.name == "c6678"
