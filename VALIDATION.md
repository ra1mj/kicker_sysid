# 实际验证记录

验证日期：2026-11-17（Asia/Tokyo）。本页记录通用合成电机和仿真台架验证，不代表任何实际电机已完成标定。

## 环境

- Python 3.12.3，Linux x86_64，NVIDIA GeForce RTX 4090。
- PyTorch 2.7.0+cu128；Warp 1.12.0+cu12；CUDA 设备实际运行。
- MuJoCo 3.8.1，mujoco-warp 3.8.1，mjlab 1.3.0，mjviser 0.0.16。
- 完整依赖及来源固定于 `uv.lock`，使用项目独立 `.venv`。

## 结果

| 检查 | 实际结果 |
|---|---|
| `pytest -q` | **24 passed，0 skipped，0 failures，0 errors** |
| Ruff | 通过 |
| `uv lock --check` | 通过 |
| wheel 与源码包 | 构建通过，含 Apache-2.0 许可证和上游来源说明 |
| `demo --backend all` | CPU、MuJoCo、mjlab/CUDA 全部通过 |
| 干净 CPU 环境 | 未安装 torch/MuJoCo，完整演示通过 |
| 干净全后端环境 | 按锁文件重新安装，`demo --backend all` 通过 |
| 编码器数据 CMA-ES 示例 | 延迟/响应时间拟合通过，验证 RMSE 为基线约 0.0000101 倍 |
| 原生 MuJoCo 生成数据后的辨识 | 惯量与基础摩擦参数误差小于 5%，留出验证通过 |
| GPU 批量隔离 | 每环境参数、不同方向、局部重置、惯量更新、张量输入/输出均通过 |

无噪声合成主演示的 R、L、Kt、Ke、转子惯量、库仑/黏性摩擦及热参数均满足 **相对误差 ≤5%**。真值文件与辨识输入分离，仅在验收阶段读取。

在独立 chirp、锁轴激励和热循环记录中，相对 CPU 的最大通道 NRMSE：

| 后端 | 最大 NRMSE | 验收阈值 |
|---|---:|---:|
| MuJoCo | 0.00073778% | 2% |
| mjlab/CUDA | 0.01728681% | 2% |

NRMSE 为 RMSE 除以该响应的范围（接近常量时采用明示的单位相关下限）。完整逐记录、逐通道结果保存在 JSON 中。

## 证据和重现

- `runs/test_results.xml`：完整测试报告。
- `runs/demo/demo_report.json`：主演示的参数恢复和平台版本。
- `runs/demo/backend_validation.json`：跨后端误差。
- `runs/demo/fit_report.json`：阶段拟合、条件数、固定参数、接受/拒绝门槛。
- `runs/demo/heldout_*.png`：已检查的验证曲线。
- `runs/clean_cpu/install_check.json`、`runs/clean_all/install_check.json`：干净环境安装结果。
- `runs/clean_all/execution.log`：干净环境三后端完整执行日志。

```bash
uv sync --locked --all-extras --group dev
uv run --no-sync pytest -q --junitxml=runs/test_results.xml
uv run --no-sync kicker-sysid demo --backend all
```

运行产物位于已忽略的 `runs/`；拷贝源码后可用以上命令重建。上游 Warp 的弃用提示未影响本次验证。
