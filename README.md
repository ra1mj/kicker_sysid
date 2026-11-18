# kicker_sysid

通用电机仿真、离线参数辨识和跨后端验证，支持 **CPU、MuJoCo、mjlab/CUDA**。可用虚拟电机完成全流程，也可导入具有明确单位和采样约定的实测日志。自带参数均为合成示例。

当前范围为电机级机电等效模型；装机关节 SysID、整车动力学、实机通信和 FOC/PWM 留待后续。

## 安装与运行

在此目录中使用 Python 3.12 和 uv，依赖锁文件为 `uv.lock`。

```bash
# CPU：不安装 torch、MuJoCo 或 CUDA
uv sync --locked
uv run --no-sync kicker-sysid demo --backend cpu

# CPU + MuJoCo
uv sync --locked --extra mujoco
uv run --no-sync kicker-sysid demo --backend mujoco

# 全部后端和测试依赖
uv sync --locked --all-extras --group dev
uv run --no-sync kicker-sysid demo --backend all
uv run --no-sync pytest -q
```

GPU 目标为 Linux x86_64 与 NVIDIA CUDA GPU。依赖固定为 mjlab 1.3.0、MuJoCo 3.8.1、mujoco-warp 3.8.1、warp-lang 1.12.0、PyTorch 2.7.0/CUDA 12.8。第一次运行 Warp 会编译内核。`--device cpu` 可显式选择 mjlab CPU；缺少请求的后端或 CUDA 时直接报错，不自动替换。

Linux GPU 库使用 PyTorch 官方 CUDA 索引，Warp 使用 NVIDIA 官方发布的 `1.12.0+cu12` 构建；来源与哈希均保存在锁文件中。

另外固定 `mjviser==0.0.16`：后续版本会引入共享 `mujoco` 模块路径的 `mujoco-core 4`，与此项目的 MuJoCo 3.8.1 不兼容。请按锁文件同步整个环境。

`uv run --no-sync` 保留已安装的可选后端；省略 `--no-sync` 时，uv 可能按默认 CPU 配置移除可选依赖。

## 常用命令

以下裸命令需要激活 `.venv`，或在前面加 `uv run --no-sync`。

```bash
kicker-sysid simulate --config configs/demo.json --backend cpu
kicker-sysid simulate --config configs/position_servo.json --backend mujoco --viewer
kicker-sysid simulate --config configs/current_drive.json --backend mjlab
kicker-sysid fit --dataset runs/demo/dataset.json
kicker-sysid validate --run runs/demo --backend all
kicker-sysid export --model runs/demo/model.json --output runs/export
python examples/identify_dynamics.py
python examples/batch_mjlab.py
```

默认无窗口运行。`--viewer` 仅用于 MuJoCo 且需要桌面显示。命令失败、验收失败、候选模型被拒绝均返回非零退出码。

## 运行产物

`demo` 默认输出到 `runs/demo/`：

| 文件 | 用途 |
|---|---|
| `dataset.json`、`data/*/trace.json`、`samples.csv` | 初始模型、实验元数据、训练/验证划分与观测日志 |
| `fit_report.json` | 参数状态、诊断、候选模型及独立验证结果 |
| `model.json` | 接受的模型；验证退化时保留基线 |
| `backend_validation.json`、`validation/*/` | 三种后端实际回放、逐记录逐通道误差 |
| `heldout_*.png` | 观测、初始模型和辨识模型的曲线比较 |
| `demo_report.json` | 参数恢复误差、验收结论、依赖版本与平台 |
| `export/` | 模型、机械台架 XML、运行脚本和 SHA-256 清单 |
| `generation_truth.json` | 演示验收真值，不传给辨识器 |

导出 XML 只描述机械台架。电气、延迟和温度必须通过 `replay.py` 或本包后端执行。导出包依赖已安装的 `kicker-sysid`，不依赖原 Sim2Motor 仓库。

## Python 接口

```python
from kicker_sysid import ActuatorSpec, Bench, simulate, fit, validate, export
from kicker_sysid.spec import demo_spec

trace = simulate(demo_spec(), Bench(load_inertia=0.0002), backend="cpu",
                 experiment={"kind": "chirp", "duration": 2.0, "amplitude": 4.0})
report = fit("runs/demo/dataset.json", output="runs/refit")
comparison = validate("runs/demo", backend="all")
export(ActuatorSpec.from_dict(report["selected_model"]), "runs/export")
```

单步接口由 `backends.make_backend()` 创建，提供 `reset()`、`observe()`、`step(Command)`、`close()`。`MJLabBackend` 另外支持 `num_envs`、每环境命令列表、`set_parameters()` 与 `reset(env_ids)`。更改惯量使用 `joint_armature`，其值只含执行器惯量；参数更新后在新实验前重置状态。

GPU 批量调用可传入 `step({"effort": tensor})`，张量为 `[num_envs, 1]` 或可广播标量；其余可用字段为 `position/velocity/external_torque/enabled`。批量输出保持为设备张量，`observe(i)` 才读取单个环境到 CPU。接入自己的 mjlab 场景时，从 `kicker_sysid.actuator` 导入 `LayeredActuatorCfg` 和 `prepare_simulation`。

## 模型与计算职责

保留 `ActuatorSpec` 和 `sim2motor.actuator.v1`。详细约定见 [模型与日志](docs/model-and-data.md)。

- 电气：RL 动态、反电动势、转矩、电压/电流限制、温漂。常数采用同一等效电压/电流约定，不自动换算线电压、相电压、峰值和 RMS。
- 传动：`N = ω_motor/ω_output`；`J_armature = N² J_rotor + J_extra`。效率作用于力矩，不乘惯量；按驱动/反驱功率方向换算。
- 台架：`Bench.load_inertia` 单独保存输出侧负载惯量。默认直驱、无重力；锁轴使用理想运动学约束。
- 摩擦：无摩擦、库仑/黏性、Stribeck；系数均为输出侧量，直驱时即电机轴侧。静摩擦由冲量或约束保持。
- 温度：铜损加热、一阶散热和电阻温漂。零电感支持代数模型；禁用驱动表示等效开路，不建模开关和续流瞬态。
- CPU 使用隐式黏性积分和干摩擦冲量；MuJoCo/mjlab 使用原生机械积分与摩擦约束。电气/驱动内核共用，转子惯量只进入一次。

电流限制通过驱动电压实现，再受母线限制。强反驱时物理电流可能超出指令限流值；模型不直接截断电流来破坏电压方程。

`controller.domain=voltage/current/torque` 的指令单位分别为电机 V、电机 A、输出 Nm。`mode=position` 在该域叠加位置 PD，误差为输出角度。延迟以物理步计，重置历史为零；控制周期必须是物理周期的整数倍。

## 辨识和数据要求

| 阶段 | 观测与实验 | 输出 |
|---|---|---|
| 电气 | 施加电压、电流、速度；锁轴动态及变速 | R、L、Ke；仅锁轴保持 Ke |
| 转矩 | 有明确定义的电磁转矩标定、电流变化 | Kt、转矩传感器偏置诊断 |
| 机械 | 电流、速度、已知台架惯量、外力矩；双向动态/滑行 | 转子惯量、库仑/黏性摩擦 |
| 热 | 电流、温度；至少 1°C 温升及加热/冷却段 | 热阻、热容 |
| 轨迹 | 指令与位置/速度等响应、显式参数边界 | Stribeck、响应时间、整数步延迟 |

基础阶段使用 SciPy 有界最小二乘并检查样本数、秩、条件数和物理可行性；扩展阶段使用 CMA-ES。直接电压驱动不能辨识电流环响应时间；Stribeck 拟合需在模型中启用，边界示例见 `configs/stribeck_bounds.json`。

```bash
# 生成编码器数据并辨识延迟/响应时间
python examples/identify_dynamics.py
kicker-sysid fit --dataset runs/dynamics/dataset.json --trajectory-bounds runs/dynamics/bounds.json
```

训练与验证按完整记录划分，拒绝重复记录。优化器不读取验证数据。最终比较整体误差及各记录各通道，通道退化时保留基线。缺少通道/激励标为 `unidentifiable`，未请求扩展参数标为 `kept`；`fitted` 描述候选估计，最终值以 `selected_model` 和参数 `selected` 字段为准。

实测导数对噪声敏感，导入前应按传感器带宽滤波、等周期重采样并保留处理记录。本包不填补缺失观测，不推测齿轮比、效率、负载惯量及通信比例。没有可信转矩标签时删除 `torque_label` 与电磁转矩列；轴端转矩须先完成惯性/摩擦补偿，不能直接用于 Kt 标定。

## 验收与维护

```bash
uv run --no-sync pytest -q
uv run --no-sync ruff check src tests examples
uv run --no-sync kicker-sysid demo --backend all
```

测试覆盖解析 RL 响应、加速/滑行/静摩擦、温升、饱和、延迟、惯量换算、数据完整性、参数恢复、独立验证、原生 MuJoCo 数据辨识、CMA-ES、张量一致性和 GPU 批量隔离。可选依赖缺失会明确跳过测试；完整验收需要全部依赖，不能用跳过结果宣称 GPU 通过。

代码来源、许可证和修改说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。
