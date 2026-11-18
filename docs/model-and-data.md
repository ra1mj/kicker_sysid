# 模型与日志约定

## 时序和坐标

每行表示 `t_k` 的状态及随后 `[t_k,t_{k+1})` 区间的驱动。位置、速度、电流、温度和电磁转矩在步进前采样。`applied_voltage_v` 是经过控制、延迟、限流及母线限制后用于本步积分的电压，不能用目标电压替代。

| 字段 | 含义 |
|---|---|
| `position_rad`、`velocity_rad_s` | 输出轴 rad、rad/s；转子速度为 N 倍 |
| `current_a` | 电机等效电流；正向电流产生正向转子转矩 |
| `electromagnetic_torque_nm` | 步进前电机侧 Kt×i；可信标定才声明 `torque_label=motor_electromagnetic` |
| `drive_torque_nm` | 本步输出侧驱动力，使用更新后的电流，包含 N 与效率，不扣摩擦 |
| `external_torque_nm` | 输出侧外部施加力矩，正数帮助正向加速 |
| `effort_command` | 控制域的 V、A 或输出 Nm；位置模式下为前馈 |
| `enabled` | 0/1；关闭为等效开路 |

`drive_torque_nm` 是诊断量，辨识器不读取它来获得真值，而从观测电流与 Kt 推导机械驱动力。轴端测力计读数不能直接替代电磁转矩。

同一编码器位置观测（偏置、量化、噪声）进入位置闭环和日志。速度噪声用于记录传感器误差，驱动反馈与反电动势使用物理速度。真实驱动器若有速度滤波，应先确定其等效响应，避免把滤波延迟解释为电机惯量。

## 方程

电气：`V = R(T)i + L di/dt + Ke N v`；本步冻结速度和温度后使用 RL 精确离散更新。

机械：`J = J_load + N² J_rotor + J_extra`，`p = J v_k + dt(τ_drive + τ_external)`，`v_{k+1} = [p − clip(p, −dt τ_dry, dt τ_dry)] / (J + dt b)`，`q_{k+1} = q_k + dt v_{k+1}`。

Stribeck 干摩擦预算为 `τ_c + τ_s exp[−(|v|/v_s)^p]`。MuJoCo/mjlab 使用原生 `frictionloss` 与 `damping`，低速粘滑不要求逐位相同。

热：`Cth dT/dt = i² R(T) − (T−Tamb)/Rth`；热阻 K/W、热容 J/K，时间常数为两者乘积。

## 导入自己的实验

每条实验一个 `trace.json` 与 CSV，或用 Python 传入含 `samples` 的字典。

```json
{
  "schema": "kicker.trace.v1",
  "name": "my_run_001",
  "csv": "samples.csv",
  "bench": {"dt": 0.001, "control_dt": 0.001, "load_inertia": 0.0002,
            "initial_position": 0.0, "initial_velocity": 0.0, "locked": false},
  "controller": {"domain": "voltage", "mode": "direct", "effort_limit": 1.0},
  "provenance": "measured; record motor, instruments, firmware and preprocessing"
}
```

CSV 至少包含 `t_s,effort_command`，添加实际测得的响应列。验证至少需要位置、速度、电流或温度之一。空白保持缺失，不能写成假零；时间须严格递增且间隔等于 `bench.dt`。初始角度/速度与记录起点一致，统一零点和转向。当前电流与温度初态是零电流、环境温度，导入时使用满足初态的完整实验。

负载力矩已知为零时明确写入 `external_torque_nm=0`，负载惯量独立获得。`dataset.json` 采用 `schema=kicker.dataset.v1`、完整 `initial_model`、`train` 与 `validation` 两组 trace.json 相对路径。编码器记录不能独立识别 Kt/R/L。

生成 CSV 带 SHA-256 校验；手工处理后重新用 `save_trace()` 更新清单并记录处理来源，读取器拒绝校验不一致的内容。

## 扩展边界

转子与台架惯量分开，为未来关节 SysID 预留接口。多体惯量、重力、接触、柔性、回差、电源回灌、温度相关 Kt、保护动作和残差网络均未作为自动辨识参数。

上游格式保留 M6 字段，但本版机械辨识与验收集中于 none/coulomb/stribeck。不能把 M6 参数解释为已验证的通用减速器特性。
