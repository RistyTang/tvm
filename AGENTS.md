# TVM 项目架构解析


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

## 目的

本项目将基于tvm框架增加对6678DSP芯片的支持。
src/target/source下完成对6678的代码生成器支持，目标为生成符合6678指令集的C代码。

# 需求

1. 生成的代码应当为void funcname(void* self_handle, void* args, int32_t num_args, void* result)格式，没有return数据