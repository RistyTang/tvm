# TVM 定制项目架构解析

本项目是基于 Apache TVM 的深度定制版本。项目将传统的 TIR（Tensor Intermediate Representation）模块拆分成了 `s_tir` 和 `tirx` 两个主要部分。

以下是项目的整体架构树状图及各目录、模块的作用分析：

```text
tvm/
├── 3rdparty/       # 存放第三方依赖库，例如 DLPack、DMLC-Core、cutlass 等
├── apps/           # 包含各种应用程序和多平台（Android/iOS等）集成部署示例
├── build/          # CMake 编译生成的对象文件和动态链接库 (如 libtvm.so)
├── ci/             # 持续集成相关的脚本和配置文件
├── cmake/          # CMake 构建系统的模块和配置 (.cmake)，控制编译选项
├── docker/         # 构建 TVM 开发与测试环境的 Docker 镜像和脚本
├── docs/           # Sphinx 源码、使用教程和 API 参考文档
├── include/        # 供外部调用的 C++ 头文件目录
│   └── tvm/
│       ├── s_tir/  # [定制] 调度与底层代码降级相关的头文件
│       └── tirx/   # [定制] TIR 基础抽象与数据结构相关的头文件
├── jvm/            # Java/Scala 前端和 JNI 绑定实现
├── python/         # TVM 的 Python 前端代码库（核心入口）
│   └── tvm/
│       ├── s_tir/  # [定制] 对应 C++ s_tir 模块的 Python API 接口
│       └── tirx/   # [定制] 对应 C++ tirx 模块的 Python API 接口
├── src/            # TVM 核心组件的 C++ 源代码目录
│   ├── arith/      # 算术分析和化简器 (分析表达式边界、证明不等式等)
│   ├── ir/         # 最基础的中间表示抽象基类 (Expr, Stmt, Module 等)
│   ├── relax/      # 新一代基于图的抽象中间表示 (Graph IR)，支持动态形状与异构编译
│   ├── runtime/    # 运行时的核心实现 (内存分配, Graph Executor, CUDA/OpenCL 等设备 API)
│   ├── script/     # TVM Script 解析器支持，实现 Python AST 到 C++ IR 对象的转换
│   ├── s_tir/      # [定制拆分] 负责 TIR 相关的调度 (Schedule)、自动调优 (Meta Schedule) 与后端代码降级
│   ├── tirx/       # [定制拆分] 负责 TIR 相关的基础结构 (AST)、内置算子、控制流与变量生命周期分析
│   ├── support/    # 通用的基础工具类 (日志系统, 内存池, 字符串处理等)
│   ├── target/     # 后端编译目标 (Target) 定义与设备专属代码生成器 (CodeGen)
│   ├── te/         # 张量表达式 (Tensor Expression)，从声明式计算公式到底层 TIR 的转换接口
│   └── topi/       # TVM 算子库 (Operator Inventory)，提供常见深度学习算子的底层实现和默认调度策略
├── tests/          # 包含 Python 和 C++ 的单元测试及集成测试
├── web/            # WebAssembly 相关的构建和绑定代码 (用于浏览器如 OPFS 环境中运行 TVM)
└── Test4dsp/       # 开发者的个人测试或特定功能的临时验证目录 (当前工作目录)
```

## 核心修改总结

相较于原版的 Apache TVM，该项目在架构上最显著的改造是对原有 `tir` 模块的解耦重构。采用这种竖向结构可以很清晰地看到分层：
1. **`tirx`**: 承载**底层的基本数据表示**（如 Expr/Stmt 数据结构抽象、AST 节点、基础的控制流分析等）。
2. **`s_tir`**: 承载**上层的调度、变换与自动调优**（如 Schedule 调度原语、Meta Schedule 搜索探索框架、Backend Lowering 降级等）。

这一纵向的职责拆分，使得 IR 的核心数据结构与针对 IR 操作的调度变换相互隔离。整体代码的耦合度降低，依赖关系更加清晰，极大地方便了各自领域的针对性维护与优化。

# 参考资料
https://juejin.cn/column/7174246303843483679
https://bytedance.larkoffice.com/docx/QP3Yd1O0HoOJABxjSm8cY0rMnce
https://tvm.hyper.ai/docs/getting-started/irmodule/
http://0fd.org/category/basic-technology/compiler/dive-into-tvm/
tvm-ffi学习（注册算子用）https://tvm.apache.org/ffi/get_started/quickstart.html
tvm-metaschedule：https://tvm.apache.org/docs/deep_dive/tensor_ir/tutorials/meta_schedule.html?utm_source=chatgpt.com
接口和字段的学习：https://tvm.apache.org/docs/reference/api/python/ir.html

# 自测学习

gcc -shared -fPIC \
    -I/home/tangqingyun/tvm/3rdparty/tvm-ffi/include \
    -I/home/tangqingyun/tvm/3rdparty/tvm-ffi/3rdparty/dlpack/include \
    src/total_test.c -o build/libtotal_test.so
