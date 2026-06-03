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
"""C6678 入口签名降级 pass（路线图 §4.8 A.7）。

为什么需要这个 pass：TVM 主 pipeline 默认的 ``MakePackedAPI`` 会把 host PrimFunc
重写为 ``int32_t __tvm_ffi_*(void* self_handle, void* args, int32_t num_args,
void* result)`` 这种 TVM FFI wrapper —— 该 wrapper 依赖 TVM runtime 才能调用，
但 C6678 BSP 端裸金属环境没有 TVM runtime，无法直接链接。

本 pass 的职责：在 ``MakePackedAPI`` 之前，对带 ``target.kind.name == "c6678"``
的 host PrimFunc 做最小改写：

1. 把 ``params`` 从原始的 ``[A_handle, B_handle, ...]``（``handle`` dtype 的 Var）
   替换为 ``[buf.data for buf in buffer_map.values()]``（``T.handle("float32", "global")``
   即 ``float*`` 这类已经带 ``PointerType`` 注解的 Var）；
2. 清空 ``buffer_map``、把 ``ret_type`` 设为 ``VoidType``；
3. 把 ``attrs["calling_conv"]`` 显式设为 ``kCPackedFunc (1)`` —— 这并不是因为我们
   真的要走 packed func，而是利用 ``MakePackedAPI::RequiresPackedAPI`` 的早退条件
   ``calling_conv != kDefault`` 让 c6678 host func 在主流水线里被跳过；
4. 把 ``attrs["tirx.is_entry_func"]`` 标上，便于后续 split host/device 与 codegen
   阶段识别这是用户可见的 bare-C 入口。

这是路径 2 的第一步（§4.8.3 步骤 1）。落地后：

* IR body 不变（``MakePackedAPI`` 之前 body 已经基于 ``Buffer.data`` 指针访问）；
* codegen 看到的 PrimFunc 直接打印成
  ``void matmul_fp32(float* A, float* B, float* C)`` 形态；
* core_mask / cfg / bias 等 BSP 形参留给 A.8 ``C6678MulticoreLower`` /
  A.9 ``C6678DMALower`` / A.10 ``dlight.c6678.Matmul`` 调度增强阶段再注入。

本 pass 必须挂在 ``default_s_tir_pipeline`` 中、``MakePackedAPI`` 之前；同时本身
对 ``target.kind.name == "c6678"`` 做守卫，其它 target 走到这里时直接原样返回。
"""

from __future__ import annotations

import tvm
from tvm.ir import CallingConv, TupleType

from .function_pass import prim_func_pass


def _is_c6678_func(func) -> bool:
    """是否是带 ``target.kind.name == "c6678"`` 的 PrimFunc。

    Parameters
    ----------
    func : tvm.tirx.PrimFunc
        待检查的 PrimFunc。

    Returns
    -------
    bool
        True 表示需要走 c6678 bare-C entry 改写；False 表示原样跳过。
    """
    if func.attrs is None:
        return False
    target = func.attrs.get("target")
    if target is None:
        return False
    return target.kind.name == "c6678"


def _already_lowered(func) -> bool:
    """避免 pass 被重复执行带来的二次改写。

    判定标准：``calling_conv`` 已经是 ``kCPackedFunc`` 或 ``kDeviceKernelLaunch``。
    这两种状态都意味着上一次 lowering 已经完成，本 pass 不应再动。
    """
    cc = None
    if func.attrs is not None:
        cc = func.attrs.get("calling_conv")
    if cc is None:
        return False
    cc_val = int(cc)
    return cc_val != int(CallingConv.DEFAULT)


@prim_func_pass(opt_level=0, name="C6678LowerEntry")
class C6678LowerEntry:
    """把 c6678 host PrimFunc 入口签名从 FFI wrapper 降级为 bare-C 形态。

    详细动机与改写规则见模块 docstring。
    """

    def transform_function(self, func, mod, ctx):  # noqa: D401
        """对单个 PrimFunc 做 entry signature 改写。

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
            改写后的 PrimFunc。若不属于 c6678 或已经 lowered，则原样返回。
        """
        del mod, ctx
        if not _is_c6678_func(func):
            return func
        if _already_lowered(func):
            return func
        if not func.buffer_map:
            # 没有 buffer_map 时无需改写（例如手写的 device kernel 形态）。
            return func

        # 1. 用 buffer.data 替换原 params；保持 params 中外部 handle 的相对顺序。
        new_params = []
        for handle_var in func.params:
            buf = func.buffer_map.get(handle_var)
            if buf is None:
                # 非 buffer 形参（例如 scalar）原样保留，让 codegen 自己打印类型。
                new_params.append(handle_var)
                continue
            new_params.append(buf.data)

        # 2. 重新构造 PrimFunc：buffer_map 清空、ret_type=void。
        new_func = tvm.tirx.PrimFunc(
            params=new_params,
            body=func.body,
            ret_type=TupleType([]),
            buffer_map={},
            attrs=func.attrs,
            span=func.span,
        )

        # 3. 显式标记 calling_conv，让后续 MakePackedAPI 直接跳过本函数。
        new_func = new_func.with_attr(
            "calling_conv", int(CallingConv.C_PACKED_FUNC)
        )
        # 4. 用 tirx.is_entry_func 标识这是用户可见的 bare-C 入口，
        #    便于后续 split host/device 与 codegen 阶段识别。
        new_func = new_func.with_attr("tirx.is_entry_func", True)
        return new_func
