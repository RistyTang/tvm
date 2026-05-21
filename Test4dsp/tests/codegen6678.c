// tvm target: {"kind":"c6678","tag":"","keys":["cpu"]}
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
int32_t test_func(void* self_handle, void* args, int32_t num_args, void* result);
int32_t test_func(void* self_handle, void* args, int32_t num_args, void* result) {
  if (!((num_args == 3))) {
    const char* __tvm_assert_parts[6] = {"Expected ", "3", " arguments", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!(!(args == NULL))) {
    const char* __tvm_assert_parts[4] = {"args pointer is NULL", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 4);
    return -1;
  }
  int32_t var_A_type_index = (((TVMFFIAny*)args)[0].type_index);
  if (!(((((var_A_type_index == 0) || (var_A_type_index == 4)) || (var_A_type_index == 7)) || (var_A_type_index >= 64)))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  void* var_A = ((var_A_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[0].v_ptr) + 24)) : (((TVMFFIAny*)args)[0].v_ptr));
  int32_t var_B_type_index = (((TVMFFIAny*)args)[1].type_index);
  if (!(((((var_B_type_index == 0) || (var_B_type_index == 4)) || (var_B_type_index == 7)) || (var_B_type_index >= 64)))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  void* var_B = ((var_B_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[1].v_ptr) + 24)) : (((TVMFFIAny*)args)[1].v_ptr));
  int32_t var_C_type_index = (((TVMFFIAny*)args)[2].type_index);
  if (!(((((var_C_type_index == 0) || (var_C_type_index == 4)) || (var_C_type_index == 7)) || (var_C_type_index >= 64)))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  void* var_C = ((var_C_type_index == 70) ? ((void*)((char*)(((TVMFFIAny*)args)[2].v_ptr) + 24)) : (((TVMFFIAny*)args)[2].v_ptr));
  if (!(!(var_A == NULL))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!((1 == (((DLTensor*)var_A)[0].ndim)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "A", ".ndim on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "1"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((((((DLTensor*)var_A)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)var_A)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)var_A)[0].dtype.lanes) == (uint16_t)1)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "A", ".dtype on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "float32"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 8);
    return -1;
  }
  void* main_var_A_shape = (((DLTensor*)var_A)[0].shape);
  int32_t n = ((int32_t)(((int64_t*)main_var_A_shape)[0]));
  void* main_var_A_strides = (((DLTensor*)var_A)[0].strides);
  if (!(((((DLTensor*)var_A)[0].device.device_type) == 1))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "A", ".device_type on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "cpu"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  int32_t dev_id = (((DLTensor*)var_A)[0].device.device_id);
  void* A = (((DLTensor*)var_A)[0].data);
  if (!(!(var_B == NULL))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!((1 == (((DLTensor*)var_B)[0].ndim)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "B", ".ndim on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "1"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((((((DLTensor*)var_B)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)var_B)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)var_B)[0].dtype.lanes) == (uint16_t)1)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "B", ".dtype on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "float32"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 8);
    return -1;
  }
  void* main_var_B_shape = (((DLTensor*)var_B)[0].shape);
  void* main_var_B_strides = (((DLTensor*)var_B)[0].strides);
  if (!(((((DLTensor*)var_B)[0].device.device_type) == 1))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "B", ".device_type on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "cpu"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  void* B = (((DLTensor*)var_B)[0].data);
  if (!(!(var_C == NULL))) {
    const char* __tvm_assert_parts[6] = {"Mismatched type on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "Tensor"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!((1 == (((DLTensor*)var_C)[0].ndim)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "C", ".ndim on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "1"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((((((DLTensor*)var_C)[0].dtype.code) == (uint8_t)2) && ((((DLTensor*)var_C)[0].dtype.bits) == (uint8_t)32)) && ((((DLTensor*)var_C)[0].dtype.lanes) == (uint16_t)1)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "C", ".dtype on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "float32"};
    TVMFFIErrorSetRaisedFromCStrParts("TypeError", __tvm_assert_parts, 8);
    return -1;
  }
  void* main_var_C_shape = (((DLTensor*)var_C)[0].shape);
  void* main_var_C_strides = (((DLTensor*)var_C)[0].strides);
  if (!(((((DLTensor*)var_C)[0].device.device_type) == 1))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "C", ".device_type on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "cpu"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  void* C = (((DLTensor*)var_C)[0].data);
  if (!(main_var_A_strides == NULL)) {
    if (!(((n == 1) || (1 == ((int32_t)(((int64_t*)main_var_A_strides)[0])))))) {
      const char* __tvm_assert_parts[7] = {"Mismatched ", "A", ".strides on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to be compact array"};
      TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 7);
      return -1;
    }
  }
  if (!(((n == 0) || !(A == NULL)))) {
    const char* __tvm_assert_parts[6] = {"A", " data pointer is NULL on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected non-NULL data pointer"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!((n == ((int32_t)(((int64_t*)main_var_B_shape)[0]))))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "B.shape[0]", " on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to match ", "A.shape[0]"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(main_var_B_strides == NULL)) {
    if (!(((n == 1) || (1 == ((int32_t)(((int64_t*)main_var_B_strides)[0])))))) {
      const char* __tvm_assert_parts[7] = {"Mismatched ", "B", ".strides on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to be compact array"};
      TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 7);
      return -1;
    }
  }
  if (!((dev_id == (((DLTensor*)var_B)[0].device.device_id)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "B.device_id", " on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to match ", "A.device_id"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((n == 0) || !(B == NULL)))) {
    const char* __tvm_assert_parts[6] = {"B", " data pointer is NULL on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected non-NULL data pointer"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!((n == ((int32_t)(((int64_t*)main_var_C_shape)[0]))))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "C.shape[0]", " on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to match ", "A.shape[0]"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(main_var_C_strides == NULL)) {
    if (!(((n == 1) || (1 == ((int32_t)(((int64_t*)main_var_C_strides)[0])))))) {
      const char* __tvm_assert_parts[7] = {"Mismatched ", "C", ".strides on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to be compact array"};
      TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 7);
      return -1;
    }
  }
  if (!((dev_id == (((DLTensor*)var_C)[0].device.device_id)))) {
    const char* __tvm_assert_parts[8] = {"Mismatched ", "C.device_id", " on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected to match ", "A.device_id"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((n == 0) || !(C == NULL)))) {
    const char* __tvm_assert_parts[6] = {"C", " data pointer is NULL on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected non-NULL data pointer"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 6);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)var_A)[0].byte_offset)))) {
    const char* __tvm_assert_parts[8] = {"Invalid ", "A.byte_offset", " on argument #", "0", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "0"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)var_B)[0].byte_offset)))) {
    const char* __tvm_assert_parts[8] = {"Invalid ", "B.byte_offset", " on argument #", "1", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "0"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  if (!(((uint64_t)0 == (((DLTensor*)var_C)[0].byte_offset)))) {
    const char* __tvm_assert_parts[8] = {"Invalid ", "C.byte_offset", " on argument #", "2", " when calling:\n  `", "main(A: Tensor([n], float32), B: Tensor([n], float32), C: Tensor([n], float32))", "`,\n  expected ", "0"};
    TVMFFIErrorSetRaisedFromCStrParts("ValueError", __tvm_assert_parts, 8);
    return -1;
  }
  int32_t i;
  for (i = 0; i < n; ++i) {
    ((float*)C)[i] = (((float*)A)[i] + ((float*)B)[i]);
  }
  return 0;
}

