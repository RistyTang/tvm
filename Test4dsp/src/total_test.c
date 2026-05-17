//自己编写一个c语言版本实现
//tvm-ffi已经提供了C ABI
#include <tvm/ffi/c_api.h>
#include <tvm/ffi/extra/c_env_api.h>

TVM_FFI_DLL_EXPORT int __tvm_ffi_add_one_c(void* handle, const TVMFFIAny* args,
                                           int32_t num_args, TVMFFIAny* result) {
  // 校验参数数量
  if (num_args != 2) {
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects exactly 2 arguments (x, y)");
    return -1;
  }
  // 提取第一个参数 x
  DLTensor* x;//3rdparty/tvm-ffi/include/tvm/ffi/c_api.h
  if (args[0].type_index == kTVMFFIDLTensorPtr) {
    x = (DLTensor*)(args[0].v_ptr);//typeless pointers
  } else if (args[0].type_index == kTVMFFITensor) {
    //TensorObj类：[ TVMFFIObject (对象头，包含引用计数等) | DLTensor (张量数据结构) ]
    x = (DLTensor*)(args[0].v_c_str + sizeof(TVMFFIObject));//raw C-string
  } else { 
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input for x"); 
    return -1; 
  }

  // 提取第二个参数 y
  DLTensor* y;
  if (args[1].type_index == kTVMFFIDLTensorPtr) {
    y = (DLTensor*)(args[1].v_ptr);
  } else if (args[1].type_index == kTVMFFITensor) {
    y = (DLTensor*)(args[1].v_c_str + sizeof(TVMFFIObject));
  } else { 
    TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input for y"); 
    return -1; 
  }

  // 核心计算逻辑：y = x + 1
  int64_t n = x->shape[0];
  float* x_data = (float*)(x->data);
  float* y_data = (float*)(y->data);

  for (int64_t i = 0; i < n; ++i) {
    y_data[i] = x_data[i] + 1.0f;
  }

  return 0; // 返回 0 表示执行成功
}