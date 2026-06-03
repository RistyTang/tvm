/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "codegen_c6678.h"

#include <iostream>
#include <tvm/arith/analyzer.h>
#include <tvm/tirx/op.h>
#include <tvm/ffi/extra/module.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/codegen.h>

#include <algorithm>
#include <string>
#include <vector>

namespace tvm {
namespace codegen {

using namespace tirx;

CodeGenC6678::CodeGenC6678() {
  // 可以进行特定的初始化
  // Init(true, "c6678");
}

void CodeGenC6678::Init(bool output_ssa, const std::string& target_str) {
  decl_stream << "// tvm target: " << target_str << "\n";
  decl_stream << "#include <math.h>\n";
  decl_stream << "#include <stdbool.h>\n";
  decl_stream << "#include <stdint.h>\n";
  decl_stream << "#include <stdio.h>\n";
  decl_stream << "#include <time.h>\n";
  decl_stream << "#include <csl/csl_cache.h>\n";
  decl_stream << "#include <csl_cacheAux.h>\n";
  decl_stream << "#include <tistdtypes.h>\n";
  decl_stream << "#include <inttypes.h>\n";
  decl_stream << "#include <stdbool.h>\n";
  decl_stream << "#include <78NE/initial.h>\n";
  decl_stream << "#include <78NE/DMA.h>\n";
  decl_stream << "#include <c_api.h>\n";
  CodeGenC::Init(output_ssa);
}

void CodeGenC6678::VisitStmt_(const ForNode* op) {
  std::string begin_str = PrintExpr(op->min);
  PrimExpr end = is_zero(op->min) ? op->extent : arith::Analyzer().Simplify(op->min + op->extent);
  std::string end_str = PrintExpr(end);
  std::string step_str = op->step.has_value() ? PrintExpr(*op->step) : "";
  
  std::string vid = AllocVarID(op->loop_var.get());
  //把循环变量拆出来声明。
  PrintIndent();
  PrintType(op->loop_var.dtype(), stream);
  stream << " " << vid << ";\n";
  
  PrintIndent();
  stream << "for (" << vid << " = " << begin_str << "; " << vid << " < " << end_str << "; ";
  if (step_str.empty()) {
    stream << "++" << vid;
  } else {
    stream << vid << " += " << step_str;
  }
  stream << ") {\n";
  int for_scope = BeginScope();
  PrintStmt(op->body);
  this->EndScope(for_scope);
  PrintIndent();
  stream << "}\n";
}

void CodeGenC6678::PrintType(DataType t, std::ostream& os) {
  int lanes = t.lanes();
  if (t.is_handle()) {
    TVM_FFI_ICHECK_EQ(lanes, 1) << "does not support vector types";
    os << "void*";
    return;
  }
  if (t.is_void()) {
    os << "void";
    return;
  }
  if (t == DataType::Bool()) {
    os << "bool";
    return;
  }
  bool fail = false;
  if (t.is_float()) {
    switch (t.bits()) {
      case 16:
        os << "half";
        break;
      case 32:
        os << "float";
        break;
      case 64:
        os << "double";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  } else if (t.is_uint() || t.is_int()) {
    if (t.is_uint()) {
      os << 'u';
    }
    switch (t.bits()) {
      case 8:
        os << "int8_t";
        break;
      case 16:
        os << "int16_t";
        break;
      case 32:
        os << "int32_t";
        break;
      case 64:
        os << "int64_t";
        break;
      case 1:
        os << "int32_t";
        break;
      default:
        fail = true;
        break;
    }
    if (!fail && lanes == 1) return;
    if (!fail && (lanes >= 2 && lanes <= 16)) {
      os << lanes;
      return;
    }
  }
  TVM_FFI_THROW(InternalError) << "Cannot convert type " << t << " to C type";
}

// 注册名为 "target.build.c6678" 的构建函数
ffi::Module BuildC6678(IRModule mod, Target target) {
  bool output_ssa = false;

  CodeGenC6678 cg;
  cg.Init(output_ssa, target->str());
  cg.SetConstantsByteAlignment(target->GetAttr<Integer>("constants-byte-alignment").value_or(16));

  auto is_aot_executor_fn = [](const PrimFunc& func) -> bool {
    return func->GetAttr<Bool>("runner_function", Bool(false)).value();
  };

  std::vector<std::pair<GlobalVar, PrimFunc>> funcs;
  for (auto [gvar, base_func] : mod->functions) {
    TVM_FFI_ICHECK(base_func->IsInstance<PrimFuncNode>()) << "CodegenC6678: Can only take PrimFunc";
    auto prim_func = Downcast<PrimFunc>(base_func);
    funcs.push_back({gvar, prim_func});
  }

  // Sort functions
  auto sort_key = [&is_aot_executor_fn](const auto& kv) {
    return std::tuple{is_aot_executor_fn(kv.second), kv.first->name_hint};
  };
  std::sort(funcs.begin(), funcs.end(), [&sort_key](const auto& kv_a, const auto& kv_b) {
    return sort_key(kv_a) < sort_key(kv_b);
  });

  ffi::Array<ffi::String> func_names;
  for (const auto& [gvar, prim_func] : funcs) {
    cg.DeclareFunction(gvar, prim_func);
  }

  // Codegen all functions
  for (const auto& [gvar, prim_func] : funcs) {
    cg.AddFunction(gvar, prim_func);
    func_names.push_back(cg.GetFunctionName(gvar));
  }

  std::string code = cg.Finish();
  // 生成为 "c6678" 格式的 Module
  return CSourceModuleCreate(code, "c6678", func_names);
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("target.build.c6678", BuildC6678);
}

}  // namespace codegen
}  // namespace tvm
