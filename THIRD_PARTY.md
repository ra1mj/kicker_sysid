# 上游代码与修改记录

本项目按 Apache License 2.0 分发，全文见 `LICENSE`。

## Sim2Motor

- 仓库：https://github.com/ra1mj/Sim2Motor
- 固定提交：`41607bea4a45e204308dfb11e02eef8b4e7fea5c`
- 来源文件：`sim/layered.py`、`sim/mjlab_actuator.py`、`LICENSE`。
- 本地位置：`src/kicker_sysid/_vendor/`。安装和运行均不依赖上游仓库。
- 本地修改：延迟缓冲区重置为零，修正首条指令绕过延迟的问题；原生摩擦初始化保留约束分配；加入测量位置偏差入口；禁用未纳入 v1 的神经网络残差入口。
- 其他台架、日志、辨识、命令行、导出和验证代码为本项目实现。阶段辨识采用 SciPy 有界最小二乘与 cmaes，不沿用上游高斯消元拟合器。

## 方法参考

- 关节执行器参数辨识：https://imchong.github.io/Robotics_Notebooks/detail.html?id=wiki-methods-joint-actuator-parameter-identification
- 对应源码：https://github.com/ImChong/Robotics_Notebooks/blob/main/wiki/methods/joint-actuator-parameter-identification.md
- MuJoCo 参数定义：https://mujoco.readthedocs.io/en/stable/XMLreference.html
- mjlab：https://github.com/mujocolab/mjlab

本项目采用分阶段激励、惯量与负载分离、完整记录留出验证的原则。没有复制参考文章的正文，也不宣称实现文章列出的所有算法。
