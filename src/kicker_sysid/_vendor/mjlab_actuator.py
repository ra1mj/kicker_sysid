"""mjlab adapter for sim2motor's layered model (optional dependencies).

Uses public ActuatorCfg/Actuator APIs and a unit-gear MuJoCo motor. Actuator
effort is already joint-side. The physics engine owns q/dq, link inertia,
gravity, contacts, dry friction and viscous damping.
"""
from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np
import torch
from mjlab.actuator.actuator import Actuator, ActuatorCfg, TransmissionType
from mjlab.managers.event_manager import requires_model_fields, RecomputeLevel
from mjlab.utils.spec import create_motor_actuator

from .layered import ActuatorKernel, ActuatorSpec, TorchOps, friction_budget


@requires_model_fields("dof_frictionloss", "dof_damping", "dof_armature")
def layered_init(env, env_ids=None):
    """Register as a startup event in ManagerBasedRlEnv configs."""
    del env, env_ids


def prepare_simulation(sim):
    """For direct Scene/Simulation use, call after scene.initialize()."""
    sim.expand_model_fields(("dof_frictionloss", "dof_damping", "dof_armature"))


def layered_joint_position(env, asset_name="robot", relative=True):
    """Policy observation in the SAME encoder frame as the drive controller.

    Use this instead of a true joint-position observation when identifying or
    randomizing layered encoder biases. Do not additionally enable mjlab's
    separate encoder-bias/action compensation for these same joints.
    """
    robot = env.scene[asset_name]
    position = robot.data.joint_pos.clone()
    for actuator in robot.actuators:
        if isinstance(actuator, LayeredActuator):
            position[:, actuator.target_ids] += actuator.kernel.parameters["encoder.bias"]
    return position - robot.data.default_joint_pos if relative else position


@dataclass(kw_only=True)
class LayeredActuatorCfg(ActuatorCfg):
    model: ActuatorSpec
    residual_network: str | None = None  # exported by calibration.learned_residual

    def __post_init__(self):
        super().__post_init__()
        if self.residual_network:
            raise ValueError("Learned residuals are outside kicker_sysid v1")
        if self.transmission_type != TransmissionType.JOINT:
            raise ValueError("Layered model currently supports rotary joints only")
        if self.delay_max_lag or self.delay_min_lag:
            raise ValueError("Set model.controller.delay_steps, not a second mjlab delay")
        if any(x is not None for x in (self.armature, self.frictionloss, self.viscous_damping)):
            raise ValueError("Armature and friction are owned by the layered model")
        if self.residual_network and (self.model.motor is not None or self.model.controller.domain != "torque"):
            raise ValueError("Residual torque networks require an ideal torque-domain base; electrical current would otherwise become inconsistent")

    def build(self, entity, target_ids, target_names):
        return LayeredActuator(self, entity, target_ids, target_names)


class LayeredActuator(Actuator):
    @property
    def command_field(self):
        # Keep this custom/stateful actuator out of mjlab's builtin fusion path.
        return None

    def edit_spec(self, spec, target_names):
        for name in target_names:
            joint = spec.joint(name)
            if joint.type != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("Motor constants are rotary: expected a hinge joint")
            if np.any(joint.stiffness):
                raise ValueError("Joint springs need an explicit load model before using this adapter")
            act = create_motor_actuator(spec, name, effort_limit=self.cfg.model.controller.effort_limit,
                gear=1.0, armature=self.cfg.model.joint_armature,
                frictionloss=max(1e-12, self.cfg.model.friction.coulomb + self.cfg.model.friction.stribeck),
                viscous_damping=self.cfg.model.friction.viscous)
            # The kernel limits effort, including per-environment parameter
            # changes. A second fixed XML clamp could silently override sysid.
            act.forcelimited = False
            act.ctrllimited = False
            self._mjs_actuators.append(act)

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._model, self._data = model, data
        self._dt = mj_model.opt.timestep
        ids = [mj_model.jnt_dofadr[self.entity.indexing.joint_ids[i].item()] for i in self._target_ids_list]
        self._dof_ids = torch.tensor(ids, dtype=torch.long, device=device)
        template = torch.zeros((data.nworld, len(ids)), device=device)
        self.kernel = ActuatorKernel(self.cfg.model, template, TorchOps())
        self.enabled = torch.ones_like(template,dtype=torch.bool)
        self._checked_fields = False
        self.residual = None
        if self.cfg.residual_network:
            from .learned import ResidualHistory
            self.residual = ResidualHistory(self.cfg.residual_network, template, self._dt)

    @staticmethod
    def _tensor(value):
        return value if isinstance(value, torch.Tensor) else value[...]

    def _fields(self):
        dry = self._tensor(self._model.dof_frictionloss)
        damping = self._tensor(self._model.dof_damping)
        if not self._checked_fields:
            n = self.kernel.zero.shape[0]
            if n > 1 and any(x.shape[0] != n or x.stride(0) == 0 for x in (dry, damping)):
                raise RuntimeError("Add layered_init as a startup event, or call prepare_simulation(sim), before stepping")
            self._checked_fields = True
        return dry, damping

    def _external_effort(self):
        """Previous-step gravity + applied + contact/limit forces, minus own dry friction."""
        bias = self._tensor(self._data.qfrc_bias)
        constraints = self._tensor(self._data.qfrc_constraint)
        applied = self._tensor(self._data.qfrc_applied)
        efc = self._data.efc
        types, indices, forces = map(self._tensor, (efc.type, efc.id, efc.force))
        valid = torch.arange(types.shape[1], device=types.device)[None, :] < self._tensor(self._data.nefc)[:, None]
        mask = valid & (types == int(mujoco.mjtConstraint.mjCNSTR_FRICTION_DOF))
        own_friction = torch.zeros_like(bias)
        own_friction.scatter_add_(1, torch.where(mask, indices, 0).long(), torch.where(mask, forces, 0.0))
        return (-bias + applied + constraints - own_friction)[:, self._dof_ids]

    def compute(self, cmd):
        dry_field, damping_field = self._fields()
        external = self._external_effort() if self.cfg.model.friction.model == "m6" else self.kernel.zero
        measured_delta = getattr(self, "measurement_delta", None)
        position = cmd.pos if measured_delta is None else cmd.pos + measured_delta
        out = self.kernel.step((cmd.position_target, cmd.velocity_target, cmd.effort_target), position, cmd.vel, self._dt, external, self.enabled)
        effort = out["effort"]
        if self.residual is not None:
            # Residual learns electromagnetic joint effort, not net effort
            # after friction, so native friction remains applied exactly once.
            delayed = self.kernel._delay_output
            error = delayed[0] - cmd.pos - self.kernel.parameters["encoder.bias"]
            effort = self.kernel.limit_effort(effort + self.residual.predict(error, cmd.vel), cmd.vel)
            effort = torch.where(self.enabled,effort,self.kernel.zero)
            out["frictionloss"], out["damping"] = friction_budget(self.kernel.parameters, self.cfg.model.friction.model, effort, external, cmd.vel, self.kernel.ops)
        dry_field[:, self._dof_ids] = out["frictionloss"]
        damping_field[:, self._dof_ids] = out["damping"]
        self.last_output = out
        return effort

    def reset(self, env_ids=None):
        super().reset(env_ids)
        if not hasattr(self, "kernel"):
            return
        mask = None
        if env_ids is not None:
            mask = torch.zeros_like(self.kernel.zero, dtype=torch.bool)
            mask[env_ids] = True
        self.kernel.reset(mask)
        if self.residual is not None:
            self.residual.reset(mask)

    def set_parameters(self, parameters, *, sim=None):
        """Assign scalar or (worlds,joints) candidates between complete rollouts.

        `joint_armature` is the TOTAL actuator-only reflected inertia. It
        requires sim so MuJoCo's constants are recomputed. Robot/link mass
        remains in the robot model. Call reset() after changing candidates.
        """
        pending = {}
        for key, value in parameters.items():
            if key != "joint_armature" and key not in self.kernel.parameters:
                raise ValueError(f"Unknown parameter {key}")
            if key.startswith("transmission.") or key in ("motor.rotor_inertia", "controller.max_delay_steps"):
                raise ValueError("Transmission is structural; use joint_armature for reflected inertia")
            t = torch.as_tensor(value, device=self.kernel.zero.device, dtype=self.kernel.zero.dtype)
            t = torch.broadcast_to(t, self.kernel.zero.shape).clone()
            if not torch.isfinite(t).all():
                raise ValueError(f"Nonfinite {key}")
            if key not in ("encoder.bias", "motor.ambient_temperature", "motor.resistance_tempco") and (t < 0).any():
                raise ValueError(f"Negative {key}")
            if key in ("motor.resistance", "motor.kt", "motor.ke", "motor.voltage_limit", "motor.current_limit", "motor.thermal_capacity", "motor.thermal_resistance", "controller.effort_limit", "controller.stall_torque", "controller.no_load_speed", "friction.stribeck_velocity", "friction.stribeck_exponent") and (t <= 0).any():
                raise ValueError(f"{key} must be positive")
            if key == "controller.delay_steps" and ((t != t.round()).any() or (t > self.cfg.model.controller.max_delay_steps).any()):
                raise ValueError("Delay must be integer physics steps within capacity")
            if key in ("controller.response_time", "controller.stall_torque", "controller.no_load_speed") and (self.kernel.parameters[key] == 0).any():
                raise ValueError(f"Enable {key} in the base model before varying it")
            pending[key] = t
        if "joint_armature" in pending:
            if sim is None:
                raise ValueError("Pass sim when changing joint_armature")
            armature = self._tensor(self._model.dof_armature)
            if armature.shape[0] != self.kernel.zero.shape[0] or (armature.shape[0] > 1 and armature.stride(0) == 0):
                raise RuntimeError("Expand dof_armature with prepare_simulation first")
            armature[:, self._dof_ids] = pending.pop("joint_armature")
            sim.recompute_constants(RecomputeLevel.set_const_0)
        self.kernel.parameters.update(pending)
