"""Unitree Go2 velocity environment configurations. File adapted from MJlab repository"""

import math
from typing import Literal

from unitree_go2.go2_constants import (
    GO2_ACTION_SCALE,
    get_go2_robot_cfg,
)

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat terrain velocity configuration."""

    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": get_go2_robot_cfg()}
    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 64
    cfg.sim.nconmax = None

    # Set terrain to plane and remove terrain generator for flat env.
    if cfg.scene.terrain is not None:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
    
    # Disable terrain curriculum.
    cfg.curriculum.pop("terrain_levels", None)

    # Remove raycast sensors and collision sensors not needed on flat.
    remove_sensors = {
        "terrain_scan",
    }
    cfg.scene.sensors = tuple(s for s in (cfg.scene.sensors or ()) if s.name not in remove_sensors)

    foot_names = ("FR", "FL", "RR", "RL")
    site_names = ("FR", "FL", "RR", "RL")
    geom_names = tuple(f"{name}_foot_collision" for name in foot_names)

    # Wire foot height scan to per-foot sites.
    for sensor in cfg.scene.sensors or ():
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(ObjRef(type="site", name=s, entity="robot") for s in site_names)
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(mode="geom", pattern=geom_names, entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )
    nonfeet_ground_cfg = ContactSensorCfg(
        name="nonfeet_ground_touch",
        primary=ContactMatch(
            mode="geom",
            entity="robot",
            # Grab all collision geoms...
            pattern=r".*_collision\d*$",
            # Except for the foot geoms.
            exclude=tuple(geom_names),
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )

    # NOTE: we are missing a trunk assembly to do self-collision with
    # self_collision_cfg = ContactSensorCfg(
    #     name="self_collision",
    #     primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    #     secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    #     fields=("found", "force"),
    #     reduce="none",
    #     num_slots=1,
    #     history_length=4,
    # )

    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        nonfeet_ground_cfg,
        # self_collision_cfg,
    )

    if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
        cfg.scene.terrain.terrain_generator.curriculum = True

    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = GO2_ACTION_SCALE

    cfg.viewer.body_name = "base_link"
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    # Replace the base foot_friction with per-axis friction events for condim 6.
    del cfg.events["foot_friction"]
    cfg.events["foot_friction_slide"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "axes": [0],
            "ranges": (0.3, 1.5),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_spin"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [1],
            "ranges": (1e-4, 2e-2),
            "shared_random": True,
        },
    )
    cfg.events["foot_friction_roll"] = EventTermCfg(
        mode="startup",
        func=envs_mdp.dr.geom_friction,
        params={
            "asset_cfg": SceneEntityCfg("robot", geom_names=geom_names),
            "operation": "abs",
            "distribution": "log_uniform",
            "axes": [2],
            "ranges": (1e-5, 5e-3),
            "shared_random": True,
        },
    )
    cfg.events["base_com"].params["asset_cfg"].body_names = ("base_link",)

    cfg.rewards["pose"].params["std_standing"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
    }
    cfg.rewards["pose"].params["std_walking"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }
    cfg.rewards["pose"].params["std_running"] = {
        r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
        r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
    }

    cfg.rewards["upright"].params["asset_cfg"].body_names = ("base_link",)
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("base_link",)

    for reward_name in ["foot_clearance", "foot_slip"]:
        cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

    cfg.rewards["body_ang_vel"].weight = 0.0
    cfg.rewards["angular_momentum"].weight = 0.0
    cfg.rewards["air_time"].weight = 0.0

    # Per-body-group collision penalties.
    # cfg.rewards["self_collisions"] = RewardTermCfg(
    #     func=mdp.self_collision_cost,
    #     weight=-0.1,
    #     params={"sensor_name": self_collision_cfg.name},
    # )
    # cfg.rewards["shank_collision"] = RewardTermCfg(
    #     func=mdp.self_collision_cost,
    #     weight=-0.1,
    #     params={"sensor_name": shank_ground_cfg.name},
    # )
    # cfg.rewards["base_link_head_collision"] = RewardTermCfg(
    #     func=mdp.self_collision_cost,
    #     weight=-0.1,
    #     params={"sensor_name": base_link_head_ground_cfg.name},
    # )

    cfg.terminations["illegal_contact"] = TerminationTermCfg(
        func=mdp.illegal_contact,
        params={"sensor_name": nonfeet_ground_cfg.name},
    )

    # On flat terrain fell_over is sufficient;
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )

    if play:
        twist_cmd = cfg.commands["twist"]
        assert isinstance(twist_cmd, UniformVelocityCommandCfg)
        twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
        twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

    return cfg
