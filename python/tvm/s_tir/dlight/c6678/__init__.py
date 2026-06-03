# isort: skip_file
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
"""C6678 DSP schedule rules.

本子包按 `s_tir/dlight/cpu` 同构组织：每个 `ScheduleRule`
通过 `is_target_available` 守卫只在 `target.kind.name == "c6678"` 时生效。

对应路线图 §4.2 的 A.5（专家模板分发器）+ A.6（已有 _p/_s 模板 schedule 化）。
当前阶段 A.5 proper 已落地：rule 入口（``Matmul``）只负责 normalize + 守卫，
schedule 主体交给 ``dispatcher.select_template`` 与 ``ScheduleTemplate`` 体系。
"""

from .dispatcher import (
    MatmulGemmTemplate,
    ScheduleTemplate,
    features_for_func,
    select_template,
)
from .elementwise import ElementGreaterEqual
from .matmul import Matmul
from .softmax import Softmax
