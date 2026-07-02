"""Hand-written MDP terms for the Unitree Go2 velocity task.

Unlike the rest of the reward/observation/event/termination library (imported
in env_cfg.py as `mdp`, from mjlab.tasks.velocity), the functions in this file
are implemented by you, using torch directly on the simulation state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def track_angular_velocity(
    env: ManagerBasedRlEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward for tracking the commanded base yaw rate (angular velocity about z).

    The commanded xy angular velocities are assumed to be zero.
    """
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    actual = asset.data.root_link_ang_vel_b

    # TODO(Exercise 8 - Angular Velocity Reward): fill in the lines below.
    # command[:, 2] = commanded yaw rate, actual[:, 2] = actual yaw rate,
    # actual[:, :2] = actual roll/pitch rate (commanded to be zero).

    z_error = None  # TODO (commanded yaw rate - actual yaw rate)^2 -> torch.square(...)
    xy_error = None  # TODO sum of squared actual roll/pitch rate -> torch.sum(torch.square(...), dim=1)
    ang_vel_error = None  # TODO z_error + xy_error

    raise NotImplementedError(
        "TODO: fill in z_error, xy_error, ang_vel_error above (see comments) and return the reward below"
    )
    return None  # TODO exp(-ang_vel_error / std^2) -> torch.exp(...)


def bad_orientation(
    env: ManagerBasedRlEnv,
    limit_angle: float,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Terminate when the robot's orientation exceeds the limit angle."""
    asset: Entity = env.scene[asset_cfg.name]
    projected_gravity = asset.data.projected_gravity_b

    # TODO(Exercise 9 - Fell Over Termination): fill in the lines below.
    # projected_gravity[:, 2] = z component of gravity in the robot's body
    # frame (close to -1 when upright, since gravity points straight down).

    tilt_angle = None  # TODO angle (rad) between "up" and the robot -> torch.acos(-projected_gravity[:, 2]).abs()

    raise NotImplementedError(
        "TODO: fill in tilt_angle above (see comments) and return the result below"
    )
    return None  # TODO True wherever tilt_angle exceeds limit_angle -> tilt_angle > limit_angle
