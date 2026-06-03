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

/*!
 * \file codegen_c6678.h
 * \brief Generate C code for TI C6678 DSP.
 */
#ifndef TVM_TARGET_SOURCE_CODEGEN_C6678_H_
#define TVM_TARGET_SOURCE_CODEGEN_C6678_H_

#include "codegen_c.h"

#include <string>

namespace tvm {
namespace codegen {

class CodeGenC6678 : public CodeGenC {
 public:
  CodeGenC6678();

  // ==============================================================
  void Init(bool output_ssa, const std::string& target_str);

  using CodeGenC::VisitStmt_;
  void VisitStmt_(const ForNode* op) override;
  void VisitStmt_(const AllocBufferNode* op) override;

  using CodeGenC::VisitExpr_;
  void VisitExpr_(const VarNode* op, std::ostream& os) override;

  using CodeGenC::PrintCallExtern;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr>& args, bool skip_first_arg,
                       std::ostream& os) override;

  using CodeGenC::PrintType;
  void PrintType(DataType t, std::ostream& os) override;
  // ==============================================================

 protected:
  void EnsureSpecialVarAlias(const VarNode* var);
};

}  // namespace codegen
}  // namespace tvm

#endif  // TVM_TARGET_SOURCE_CODEGEN_C6678_H_
