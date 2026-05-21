import tvm
from tvm import te
import os

# 1. 定义一个简单的张量加法计算
n = te.var("n")
A = te.placeholder((n,), name='A')
B = te.placeholder((n,), name='B')
C = te.compute(A.shape, lambda i: A[i] + B[i], name='C')

# 在比较新的 TVM 且不用 MetaSchedule 时，可以直接用 te.create_prim_func 配合 tvm.build
func = te.create_prim_func([A, B, C])

# 2. 核心！将 Target 指定为 6678 C 语言
target = tvm.target.Target("c6678")
print("Target kind name is:", target.kind.name)

# 3. 编译模型，直接传入 prim_func
mod = tvm.build(func, target=target)

# 4. 获取并打印生成的 C99 代码。传入 "c6678" 格式
c_source_code = mod.inspect_source("c6678")
print("================ Generated C6678 Code ================")
print(c_source_code)

# 5. 你也可以把它保存到文件中，然后拖到你的 CCS 工程里编译
output_dir = os.path.join(os.path.dirname(__file__), "tests")
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "my_vector_add.c")

with open(output_file, "w") as f:
    f.write(c_source_code)
print(f"Generated C code written to {output_file}")