"""Real mjlab Scene/Simulation adapter. No emulation or backend fallback."""

from .backends import bench_xml
from .signals import Command


class MJLabBackend:
    name = "mjlab"

    def __init__(self, spec, bench, *, num_envs=1, device="cuda:0"):
        try:
            import mujoco
            import torch
            from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
            from mjlab.scene import Scene, SceneCfg
            from mjlab.sim import MujocoCfg, Simulation, SimulationCfg

            from ._vendor.mjlab_actuator import LayeredActuatorCfg, prepare_simulation
        except ImportError as exc:
            raise RuntimeError("mjlab backend is unavailable; install kicker-sysid[mjlab]") from exc
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; no automatic CPU fallback")
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.torch, self.device = torch, device
        self.spec, self.bench, self.num_envs = spec, bench, num_envs
        cfg = LayeredActuatorCfg(target_names_expr=("shaft",), model=spec)
        xml = bench_xml(spec, bench, actuator=False)
        entity = EntityCfg(
            spec_fn=lambda: mujoco.MjSpec.from_string(xml),
            articulation=EntityArticulationInfoCfg(actuators=(cfg,)),
        )
        self.scene = Scene(SceneCfg(num_envs=num_envs, entities={"motor": entity}), device=device)
        self.sim = Simulation(
            num_envs=num_envs,
            cfg=SimulationCfg(
                mujoco=MujocoCfg(
                    timestep=bench.dt, integrator="implicitfast", gravity=(0, 0, 0), tolerance=1e-10
                )
            ),
            model=self.scene.compile(),
            device=device,
        )
        self.scene.initialize(self.sim.mj_model, self.sim.model, self.sim.data)
        prepare_simulation(self.sim)
        self.robot = self.scene["motor"]
        self.actuator = self.robot.actuators[0]
        self.kernel = self.actuator.kernel
        self.reset()

    def _values(self, values):
        return self.torch.as_tensor(values, device=self.device, dtype=self.kernel.zero.dtype).reshape(-1, 1)

    def reset(self, env_ids=None):
        torch = self.torch
        ids = (
            torch.arange(self.num_envs, device=self.device)
            if env_ids is None
            else torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        )
        if (ids < 0).any() or (ids >= self.num_envs).any():
            raise ValueError("Invalid environment id")
        self.sim.reset(ids)
        self.scene.reset(ids)
        q = torch.full((len(ids), 1), self.bench.initial_position, device=self.device)
        v = torch.full_like(q, 0.0 if self.bench.locked else self.bench.initial_velocity)
        self.robot.write_joint_state_to_sim(q, v, env_ids=ids)
        self.actuator.enabled[ids] = True
        self.sim.data.qfrc_applied[ids] = 0
        self.sim.forward()
        self.scene.update(self.bench.dt)

    def observe(self, env=0):
        if not 0 <= env < self.num_envs:
            raise IndexError(env)
        return dict(
            position=self.robot.data.joint_pos[env, 0].item(),
            velocity=self.robot.data.joint_vel[env, 0].item(),
            current=self.kernel.current[env, 0].item() if self.spec.motor else None,
            temperature=self.kernel.temperature[env, 0].item() if self.spec.motor else None,
        )

    def step(self, command, measured_position=None):
        if isinstance(command, dict):
            unknown = set(command) - {"position", "velocity", "effort", "external_torque", "enabled"}
            if unknown:
                raise ValueError(f"Unknown batched command fields: {sorted(unknown)}")
            values = {}
            for name in ("position", "velocity", "effort", "external_torque", "enabled"):
                tensor = self._values(command.get(name, True if name == "enabled" else 0.0))
                values[name] = self.torch.broadcast_to(tensor, (self.num_envs, 1))
        else:
            commands = [command] * self.num_envs if isinstance(command, Command) else list(command)
            if len(commands) != self.num_envs:
                raise ValueError("One command per environment is required")
            values = {
                name: self._values([getattr(c, name) for c in commands])
                for name in ("position", "velocity", "effort", "external_torque", "enabled")
            }
        self.robot.set_joint_position_target(values["position"])
        self.robot.set_joint_velocity_target(values["velocity"])
        self.robot.set_joint_effort_target(values["effort"])
        self.actuator.enabled[:] = values["enabled"].bool()
        self.actuator.measurement_delta = (
            None
            if measured_position is None
            else (
                self._values(
                    [measured_position] if isinstance(measured_position, (int, float)) else measured_position
                )
                - self.robot.data.joint_pos
                - self.kernel.parameters["encoder.bias"]
            )
        )
        self.scene.write_data_to_sim()
        self.sim.data.qfrc_applied[:, self.actuator._dof_ids] = values["external_torque"]
        self.sim.step()
        if self.bench.locked:
            q = self.torch.full((self.num_envs, 1), self.bench.initial_position, device=self.device)
            self.robot.write_joint_state_to_sim(q, self.torch.zeros_like(q))
            self.sim.forward()
        self.scene.update(self.bench.dt)
        if not self.torch.isfinite(self.robot.data.joint_pos).all():
            raise RuntimeError("mjlab produced nonfinite positions")
        out = self.actuator.last_output
        # Keep batch output on-device; callers choose when to copy observations.
        if self.num_envs > 1:
            return out
        return {k: (v[0, 0].item() if v is not None else None) for k, v in out.items()}

    def set_parameters(self, parameters):
        self.actuator.set_parameters(parameters, sim=self.sim)

    def close(self):
        pass
