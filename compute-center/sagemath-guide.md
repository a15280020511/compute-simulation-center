# SageMath受控符号计算能力

- 生产操作：`symbolic_mathematics`
- 模式：`simplify`、`solve`、`differentiate`、`integrate`、`matrix_analysis`、`number_theory`
- 版本：SageMath 10.9
- 镜像：`sagemath/sagemath@sha256:e068670ae5863b54b2550e72437ec637b0283acb0dc712c8584c124dbf44e667`
- 网络：镜像预拉取完成后，执行时同时受计算中心外层网络命名空间断网和Docker `--network none`约束。
- 文件系统：只读根文件系统，仅只读挂载固定运行脚本与结构化输入；临时目录为无执行权限的tmpfs。
- 权限：移除全部Linux capabilities，启用`no-new-privileges`。
- 输入：不接受任意Python或Sage代码，只接受受限表达式语法、变量和函数白名单，以及有界矩阵或整数数组。
- 资源：90秒、3072MB、2 CPU、128 PID。
- 模型/API调用：0。
- 回滚：撤销注册表和能力文件提交，不影响现有29项计算操作。
