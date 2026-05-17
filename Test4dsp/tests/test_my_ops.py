import ctypes
import numpy as np
import tvm_ffi
from tvm_ffi import libinfo

def test_my_custom_add():
    # 1. 加载你刚刚编译好的动态链接库
    lib = libinfo.load_lib_ctypes("libtotal_test.so")
    
    # 2. 从库中获取你的算子函数
    # 注意：在 Python 侧调用时，不需要加上 __tvm_ffi_ 前缀
    my_add_func = lib.get_func("add_one_c")
    
    # 3. 准备测试数据
    # 因为你用的是 C ABI，它支持任何实现了 DLPack 协议的张量（如 numpy, PyTorch）
    x = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    y = np.empty_like(x)
    
    # 4. 执行算子
    my_add_func(x, y)
    
    # 5. 验证结果
    np.testing.assert_allclose(y, x + 1.0)
    print("Test passed!")

if __name__ == "__main__":
    test_my_custom_add()