#!/usr/bin/env python3
"""Render randomized UR5e block-stacking episodes for the website."""

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import zlib

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = ROOT / "assets" / "simulation"
DEFAULT_MENAGERIE = Path(
    os.environ.get("MUJOCO_MENAGERIE", "/tmp/mujoco_menagerie")
)

FPS = 60
WIDTH = 1920
HEIGHT = 1216
BLOCK_HALF = 0.035
GRASP_AMOUNT = 0.17
STACK_LEFT = np.array([-0.16, -0.04])
STACK_RIGHT = np.array([0.16, -0.04])
BLOCK_NAMES = ("red", "blue", "yellow", "green")

ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)

OPEN_FINGERS = np.array(
    [0.003, 0.0, 0.003, -0.003, 0.003, 0.0, 0.003, -0.003]
)
CLOSED_FINGERS = np.array(
    [0.782, 0.0, 0.781, -0.758, 0.782, 0.0, 0.781, -0.758]
)

# A deliberately imperfect home stack. Every episode returns to this exact pose,
# so switching between randomized rollouts remains seamless.
LEFT_STACK_LAYOUT = {
    "red": (
        np.array([-0.162, -0.038, BLOCK_HALF]),
        np.array([0.008, -0.006, -0.055]),
    ),
    "blue": (
        np.array([-0.157, -0.043, BLOCK_HALF * 3 + 0.001]),
        np.array([-0.010, 0.007, 0.076]),
    ),
    "yellow": (
        np.array([-0.164, -0.039, BLOCK_HALF * 5 + 0.002]),
        np.array([0.007, 0.011, -0.091]),
    ),
    "green": (
        np.array([-0.158, -0.044, BLOCK_HALF * 7 + 0.003]),
        np.array([-0.009, -0.006, 0.047]),
    ),
}


SCENE_XML = """
<mujoco model="ur5e randomized block stacking">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" integrator="implicitfast" gravity="0 0 -9.81"/>
  <visual>
    <global azimuth="90" elevation="-20" offwidth="1920" offheight="1216"/>
    <quality shadowsize="4096" offsamples="4"/>
    <map znear="0.01" zfar="8" fogstart="3" fogend="6"/>
    <headlight ambient="0.48 0.47 0.46" diffuse="0.32 0.32 0.33" specular="0.14 0.14 0.15"/>
    <rgba haze="0.957 0.945 0.918 1"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.957 0.945 0.918" rgb2="0.957 0.945 0.918" width="512" height="3072"/>
    <texture name="table_tex" type="2d" builtin="flat" rgb1="0.80 0.79 0.81" width="64" height="64"/>
    <material name="table_mat" texture="table_tex" rgba="1 1 1 1" roughness="0.82"/>
    <material name="edge_mat" rgba="0.52 0.50 0.54 1" roughness="0.68"/>
    <material name="red" rgba="0.75 0.40 0.35 1" roughness="0.56"/>
    <material name="blue" rgba="0.44 0.33 0.53 1" roughness="0.56"/>
    <material name="yellow" rgba="0.79 0.59 0.24 1" roughness="0.56"/>
    <material name="green" rgba="0.42 0.55 0.44 1" roughness="0.56"/>
  </asset>
  <worldbody>
    <light pos="0 -1.8 3.0" dir="0 0.35 -1" directional="true" diffuse="0.34 0.33 0.32" specular="0.13 0.13 0.14" castshadow="true"/>
    <light pos="-2 -0.4 1.7" dir="1 0 -0.5" directional="true" diffuse="0.10 0.11 0.14" specular="0.05 0.06 0.08"/>
    <body name="table" pos="0 0 -0.075">
      <geom name="table_top" type="box" size="1.08 0.72 0.075" material="table_mat"/>
      <geom type="box" pos="0 0 -0.083" size="1.09 0.73 0.008" material="edge_mat" contype="0" conaffinity="0"/>
    </body>
    <body name="red_block" mocap="true" pos="-0.162 -0.038 0.035">
      <geom name="red_block_geom" type="box" size="0.035 0.035 0.035" material="red"/>
    </body>
    <body name="blue_block" mocap="true" pos="-0.157 -0.043 0.105">
      <geom name="blue_block_geom" type="box" size="0.035 0.035 0.035" material="blue"/>
    </body>
    <body name="yellow_block" mocap="true" pos="-0.164 -0.039 0.175">
      <geom name="yellow_block_geom" type="box" size="0.035 0.035 0.035" material="yellow"/>
    </body>
    <body name="green_block" mocap="true" pos="-0.158 -0.044 0.245">
      <geom name="green_block_geom" type="box" size="0.035 0.035 0.035" material="green"/>
    </body>
  </worldbody>
</mujoco>
"""


def arm_with_gripper(menagerie: Path) -> mujoco.MjSpec:
    arm_path = menagerie / "universal_robots_ur5e" / "ur5e.xml"
    gripper_path = menagerie / "robotiq_2f85" / "2f85.xml"
    if not arm_path.exists() or not gripper_path.exists():
        raise FileNotFoundError(
            "Expected universal_robots_ur5e and robotiq_2f85 under "
            f"{menagerie}. See tools/simulation/README.md."
        )

    arm = mujoco.MjSpec.from_file(str(arm_path))
    gripper = mujoco.MjSpec.from_file(str(gripper_path))
    arm.attach(gripper, site="attachment_site", prefix="gripper_")
    for key in list(arm.keys):
        arm.delete(key)
    for light in list(arm.lights):
        arm.delete(light)
    return arm


def build_model(menagerie: Path) -> mujoco.MjModel:
    scene = mujoco.MjSpec.from_string(SCENE_XML)
    left_frame = scene.worldbody.add_frame(
        pos=[-0.56, 0.08, 0.0], euler=[0, 0, -1.5708]
    )
    scene.attach(
        arm_with_gripper(menagerie), frame=left_frame, prefix="left_"
    )
    scene.compile()

    right_frame = scene.worldbody.add_frame(
        pos=[0.56, 0.08, 0.0], euler=[0, 0, 1.5708]
    )
    scene.attach(
        arm_with_gripper(menagerie), frame=right_frame, prefix="right_"
    )
    return scene.compile()


def smootherstep(value: float) -> float:
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (value * (value * 6 - 15) + 10)


def motion_progress(
    progress: float,
    start_at_rest: bool,
    end_at_rest: bool,
    ramp: float = 0.12,
) -> float:
    start_ramp = ramp if start_at_rest else 0.0
    end_ramp = ramp if end_at_rest else 0.0
    total_area = 1.0 - (start_ramp + end_ramp) / 2

    if start_ramp and progress < start_ramp:
        distance = progress**2 / (2 * start_ramp)
    elif end_ramp and progress > 1.0 - end_ramp:
        constant_area = start_ramp / 2 + 1.0 - end_ramp - start_ramp
        elapsed = progress - (1.0 - end_ramp)
        distance = constant_area + elapsed - elapsed**2 / (2 * end_ramp)
    else:
        distance = start_ramp / 2 + progress - start_ramp

    return float(np.clip(distance / total_area, 0.0, 1.0))


def orientation_matrix(orientation: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = orientation
    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cosine_roll, -sine_roll], [0.0, sine_roll, cosine_roll]]
    )
    rotation_y = np.array(
        [
            [cosine_pitch, 0.0, sine_pitch],
            [0.0, 1.0, 0.0],
            [-sine_pitch, 0.0, cosine_pitch],
        ]
    )
    rotation_z = np.array(
        [[cosine_yaw, -sine_yaw, 0.0], [sine_yaw, cosine_yaw, 0.0], [0.0, 0.0, 1.0]]
    )
    return rotation_z @ rotation_y @ rotation_x


def orientation_quaternion(orientation: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = orientation / 2
    cosine_roll, sine_roll = np.cos(roll), np.sin(roll)
    cosine_pitch, sine_pitch = np.cos(pitch), np.sin(pitch)
    cosine_yaw, sine_yaw = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            cosine_roll * cosine_pitch * cosine_yaw
            + sine_roll * sine_pitch * sine_yaw,
            sine_roll * cosine_pitch * cosine_yaw
            - cosine_roll * sine_pitch * sine_yaw,
            cosine_roll * sine_pitch * cosine_yaw
            + sine_roll * cosine_pitch * sine_yaw,
            cosine_roll * cosine_pitch * sine_yaw
            - sine_roll * sine_pitch * cosine_yaw,
        ]
    )


def clipped_sample(
    rng: np.random.Generator,
    center: np.ndarray,
    sigma: np.ndarray,
    limit: np.ndarray,
) -> np.ndarray:
    noise = np.clip(rng.normal(0.0, sigma), -limit, limit)
    return center + noise


class IKSolver:
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, prefix: str):
        self.model = model
        self.data = data
        self.site_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SITE, f"{prefix}gripper_pinch"
        )
        joint_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}{joint_name}"
            )
            for joint_name in ARM_JOINTS
        ]
        self.qpos_ids = np.array(
            [model.jnt_qposadr[joint_id] for joint_id in joint_ids]
        )
        self.dof_ids = np.array(
            [model.jnt_dofadr[joint_id] for joint_id in joint_ids]
        )
        self.jac_pos = np.zeros((3, model.nv))
        self.jac_rot = np.zeros((3, model.nv))
        mujoco.mj_forward(model, data)
        self.base_rotation = data.site_xmat[self.site_id].reshape(3, 3).copy()
        self.current_orientation = np.zeros(3)

    def position(self) -> np.ndarray:
        mujoco.mj_forward(self.model, self.data)
        return self.data.site_xpos[self.site_id].copy()

    def solve(
        self,
        target_position: np.ndarray,
        target_orientation: np.ndarray,
        iterations: int = 80,
    ) -> float:
        target_rotation = (
            orientation_matrix(target_orientation) @ self.base_rotation
        )
        for _ in range(iterations):
            mujoco.mj_forward(self.model, self.data)
            current_rotation = self.data.site_xmat[self.site_id].reshape(3, 3)
            position_error = target_position - self.data.site_xpos[self.site_id]
            rotation_error = 0.5 * sum(
                np.cross(current_rotation[:, axis], target_rotation[:, axis])
                for axis in range(3)
            )
            if (
                np.linalg.norm(position_error) < 2e-5
                and np.linalg.norm(rotation_error) < 2e-4
            ):
                break

            error = np.concatenate([position_error, 0.45 * rotation_error])
            mujoco.mj_jacSite(
                self.model,
                self.data,
                self.jac_pos,
                self.jac_rot,
                self.site_id,
            )
            jacobian = np.vstack(
                [
                    self.jac_pos[:, self.dof_ids],
                    0.45 * self.jac_rot[:, self.dof_ids],
                ]
            )
            damping = 2e-3
            delta = jacobian.T @ np.linalg.solve(
                jacobian @ jacobian.T + damping * np.eye(6), error
            )
            self.data.qpos[self.qpos_ids] += np.clip(delta, -0.035, 0.035)

        self.current_orientation = target_orientation.copy()
        mujoco.mj_forward(self.model, self.data)
        return float(
            np.linalg.norm(target_position - self.data.site_xpos[self.site_id])
        )


def hermite_value(
    start: np.ndarray,
    end: np.ndarray,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    duration: float,
    parameter: float,
) -> np.ndarray:
    parameter_squared = parameter * parameter
    parameter_cubed = parameter_squared * parameter
    return (
        (2 * parameter_cubed - 3 * parameter_squared + 1) * start
        + (parameter_cubed - 2 * parameter_squared + parameter)
        * duration
        * start_tangent
        + (-2 * parameter_cubed + 3 * parameter_squared) * end
        + (parameter_cubed - parameter_squared) * duration * end_tangent
    )


class Timeline:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        rng: np.random.Generator,
    ):
        self.model = model
        self.data = data
        self.rng = rng
        self.solvers = {
            "left": IKSolver(model, data, "left_"),
            "right": IKSolver(model, data, "right_"),
        }
        self.grip = {"left": 0.0, "right": 0.0}
        self.blocks = {
            name: position.copy()
            for name, (position, _) in LEFT_STACK_LAYOUT.items()
        }
        self.block_orientations = {
            name: orientation.copy()
            for name, (_, orientation) in LEFT_STACK_LAYOUT.items()
        }
        self.frames = []
        self.max_ik_error = 0.0
        self.unsafe_contacts = set()

    def finger_pose(self, amount: float) -> np.ndarray:
        return OPEN_FINGERS + (CLOSED_FINGERS - OPEN_FINGERS) * amount

    def geom_label(self, geom_id: int) -> str:
        name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if name:
            return name
        body_id = self.model.geom_bodyid[geom_id]
        body_name = mujoco.mj_id2name(
            self.model, mujoco.mjtObj.mjOBJ_BODY, body_id
        )
        return f"{body_name}:collision"

    def append(self, attached: tuple[str, str] | None = None) -> None:
        self.data.qpos[6:14] = self.finger_pose(self.grip["left"])
        self.data.qpos[20:28] = self.finger_pose(self.grip["right"])
        mujoco.mj_forward(self.model, self.data)

        if attached is not None:
            arm, block = attached
            self.blocks[block] = self.solvers[arm].position()

        for name in BLOCK_NAMES:
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_block"
            )
            mocap_id = self.model.body_mocapid[body_id]
            self.data.mocap_pos[mocap_id] = self.blocks[name]
            self.data.mocap_quat[mocap_id] = orientation_quaternion(
                self.block_orientations[name]
            )

        mujoco.mj_forward(self.model, self.data)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            if contact.dist >= -1e-4:
                continue
            geom1 = self.geom_label(contact.geom1)
            geom2 = self.geom_label(contact.geom2)
            if "_block_geom" not in geom1 and "_block_geom" not in geom2:
                self.unsafe_contacts.add(
                    (geom1, geom2, round(float(contact.dist), 5))
                )

        self.frames.append(
            (
                self.data.qpos.copy(),
                {name: self.blocks[name].copy() for name in BLOCK_NAMES},
                {
                    name: self.block_orientations[name].copy()
                    for name in BLOCK_NAMES
                },
            )
        )

    def hold(self, duration: float) -> None:
        for _ in range(max(1, round(duration * FPS))):
            self.append()

    def interpolated_grip(
        self,
        progress: float,
        keyframes: list[tuple[float, float]],
    ) -> float:
        if progress <= keyframes[0][0]:
            return keyframes[0][1]
        for start, end in zip(keyframes, keyframes[1:]):
            if progress <= end[0]:
                parameter = (progress - start[0]) / (end[0] - start[0])
                return start[1] + (end[1] - start[1]) * smootherstep(
                    parameter
                )
        return keyframes[-1][1]

    def decision_positions(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_lift: float,
        air_lift: float,
        end_drop: float,
        bend_sigma: float = 0.024,
        bend_limit: float = 0.045,
    ) -> list[np.ndarray]:
        displacement = end - start
        planar_displacement = displacement[:2]
        planar_distance = float(np.linalg.norm(planar_displacement))
        if planar_distance > 1e-9:
            normal = np.array(
                [-planar_displacement[1], planar_displacement[0]]
            ) / planar_distance
        else:
            normal = np.array([0.0, 1.0])

        route_bend = float(
            np.clip(
                self.rng.normal(0.0, bend_sigma),
                -bend_limit,
                bend_limit,
            )
        )
        first_bend = float(
            np.clip(
                route_bend + self.rng.normal(0.0, bend_sigma * 0.45),
                -bend_limit,
                bend_limit,
            )
        )
        second_bend = float(
            np.clip(
                0.65 * route_bend
                + self.rng.normal(0.0, bend_sigma * 0.55),
                -bend_limit,
                bend_limit,
            )
        )
        departure_fraction = float(self.rng.uniform(0.09, 0.16))
        first_air_fraction = float(self.rng.uniform(0.31, 0.46))
        second_air_fraction = float(self.rng.uniform(0.58, 0.75))
        approach_fraction = float(self.rng.uniform(0.86, 0.93))
        departure = start + displacement * departure_fraction
        first_air = start + displacement * first_air_fraction
        second_air = start + displacement * second_air_fraction
        approach = start + displacement * approach_fraction
        departure[:2] += normal * route_bend * 0.25
        first_air[:2] += normal * first_bend
        second_air[:2] += normal * second_bend
        approach[:2] += normal * second_bend * 0.35
        departure[2] = start[2] + start_lift
        first_air[2] = (
            start[2]
            + displacement[2] * first_air_fraction
            + air_lift * float(self.rng.uniform(0.88, 1.16))
        )
        second_air[2] = (
            start[2]
            + displacement[2] * second_air_fraction
            + air_lift * float(self.rng.uniform(0.68, 1.08))
        )
        approach[2] = end[2] + end_drop
        return [departure, first_air, second_air, approach]

    def state_orientations(
        self,
        start: np.ndarray,
        end: np.ndarray,
        fractions: tuple[float, ...] = (0.13, 0.38, 0.67, 0.89),
        sigma: float = 0.018,
        limit: float = 0.032,
    ) -> list[np.ndarray]:
        displacement = end - start
        orientations = []
        for fraction in fractions:
            orientation = start + displacement * fraction
            orientation += np.clip(
                self.rng.normal(0.0, sigma, 3), -limit, limit
            )
            orientations.append(orientation)
        return orientations

    def endpoint_clearance(self, position: np.ndarray) -> float:
        layer = float(
            np.clip(
                (position[2] - BLOCK_HALF)
                / (6 * BLOCK_HALF + 0.003),
                0.0,
                1.0,
            )
        )
        center = 0.021 + 0.009 * layer
        return float(
            np.clip(center + self.rng.normal(0.0, 0.003), 0.018, 0.034)
        )

    def placement_layer(self, position: np.ndarray) -> float:
        return float(
            np.clip(
                (position[2] - BLOCK_HALF)
                / (6 * BLOCK_HALF + 0.003),
                0.0,
                1.0,
            )
        )

    def sampled_midair_positions(
        self,
        start: np.ndarray,
        end: np.ndarray,
        start_lift: float,
        end_drop: float,
    ) -> list[np.ndarray]:
        departure = start.copy()
        departure[2] += start_lift

        planar_displacement = end[:2] - start[:2]
        planar_distance = float(np.linalg.norm(planar_displacement))
        if planar_distance > 1e-9:
            normal = np.array(
                [-planar_displacement[1], planar_displacement[0]]
            ) / planar_distance
        else:
            normal = np.array([0.0, 1.0])
        midpoint_fraction = float(self.rng.uniform(0.30, 0.70))
        midpoint = start + (end - start) * midpoint_fraction
        maximum_bend = min(0.045, 0.25 * planar_distance)
        minimum_bend = min(0.012, maximum_bend)
        bend = float(self.rng.uniform(minimum_bend, maximum_bend))
        midpoint[:2] += normal * bend * self.rng.choice((-1.0, 1.0))
        layer = self.placement_layer(end)
        air_clearance = float(self.rng.uniform(0.032, 0.054)) + 0.012 * layer
        midpoint[2] = max(
            max(start[2], end[2]) + air_clearance,
            departure[2] + 0.008,
            end[2] + end_drop + 0.008,
        )

        approach = end.copy()
        approach[2] += end_drop
        return [departure, midpoint, approach]

    def sampled_grasp_approach(
        self, source: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        layer = self.placement_layer(source)
        clearance = float(self.rng.uniform(0.052, 0.075)) + 0.006 * layer
        radius = float(self.rng.uniform(0.008, min(0.026, 0.42 * clearance)))
        azimuth = float(self.rng.uniform(0.0, 2 * np.pi))
        lateral_offset = np.array(
            [radius * np.cos(azimuth), radius * np.sin(azimuth)]
        )
        pregrasp = source.copy()
        pregrasp[:2] += lateral_offset
        pregrasp[2] += clearance
        aligned = source.copy()
        aligned[:2] += lateral_offset * float(self.rng.uniform(0.12, 0.24))
        aligned[2] += float(self.rng.uniform(0.038, 0.048))
        return pregrasp, aligned

    def sampled_reach_bend(
        self, start: np.ndarray, end: np.ndarray
    ) -> np.ndarray:
        displacement = end - start
        planar_displacement = displacement[:2]
        planar_distance = float(np.linalg.norm(planar_displacement))
        if planar_distance > 1e-9:
            normal = np.array(
                [-planar_displacement[1], planar_displacement[0]]
            ) / planar_distance
        else:
            normal = np.array([0.0, 1.0])
        fraction = float(self.rng.uniform(0.30, 0.70))
        bend = start + displacement * fraction
        maximum_bend = min(0.045, 0.25 * planar_distance)
        if maximum_bend > 0.008:
            magnitude = float(self.rng.uniform(0.008, maximum_bend))
            bend[:2] += normal * magnitude * self.rng.choice((-1.0, 1.0))
        bend[2] += float(self.rng.uniform(-0.010, 0.030))
        return bend

    def move_state_path(
        self,
        arm: str,
        states: list[np.ndarray],
        end_position: np.ndarray,
        orientation_states: list[np.ndarray],
        end_orientation: np.ndarray,
        cruise_speed: float = 0.21,
        attached_block: str | None = None,
        grip_keyframes: list[tuple[float, float]] | None = None,
        start_at_rest: bool = True,
        end_at_rest: bool = True,
        vertical_departure: bool = False,
        vertical_approach: bool = False,
        approach_slowdown_distance: float = 0.0,
        approach_speed_scale: float = 1.0,
        endpoint_ramp: float = 0.12,
    ) -> None:
        solver = self.solvers[arm]
        start_position = solver.position()
        start_orientation = solver.current_orientation.copy()
        position_values = np.asarray(
            [start_position, *states, end_position], dtype=float
        )
        orientation_values = np.asarray(
            [start_orientation, *orientation_states, end_orientation],
            dtype=float,
        )
        segment_lengths = np.linalg.norm(
            np.diff(position_values, axis=0), axis=1
        ) + 0.06 * np.linalg.norm(
            np.diff(orientation_values, axis=0), axis=1
        )
        segment_lengths = np.maximum(segment_lengths, 1e-4)
        knot_times = np.concatenate([[0.0], np.cumsum(segment_lengths)])
        knot_times /= knot_times[-1]

        tangent_scale = float(self.rng.uniform(0.72, 0.82))

        def path_tangents(
            values: np.ndarray,
        ) -> np.ndarray:
            tangents = np.zeros_like(values)
            for index in range(1, len(values) - 1):
                span = knot_times[index + 1] - knot_times[index - 1]
                tangents[index] = (
                    tangent_scale
                    * (values[index + 1] - values[index - 1])
                    / span
                )
            if not start_at_rest:
                tangents[0] = (
                    tangent_scale
                    * (values[1] - values[0])
                    / (knot_times[1] - knot_times[0])
                )
            if not end_at_rest:
                tangents[-1] = (
                    tangent_scale
                    * (values[-1] - values[-2])
                    / (knot_times[-1] - knot_times[-2])
                )
            return tangents

        position_tangents = path_tangents(position_values)
        orientation_tangents = path_tangents(orientation_values)
        if vertical_departure:
            position_tangents[1, :2] = 0.0
        if vertical_approach:
            position_tangents[-2, :2] = 0.0
        dense_positions = [position_values[0]]
        dense_orientations = [orientation_values[0]]
        for segment in range(len(position_values) - 1):
            segment_duration = knot_times[segment + 1] - knot_times[segment]
            for index in range(1, 33):
                parameter = index / 32
                dense_positions.append(
                    hermite_value(
                        position_values[segment],
                        position_values[segment + 1],
                        position_tangents[segment],
                        position_tangents[segment + 1],
                        segment_duration,
                        parameter,
                    )
                )
                dense_orientations.append(
                    hermite_value(
                        orientation_values[segment],
                        orientation_values[segment + 1],
                        orientation_tangents[segment],
                        orientation_tangents[segment + 1],
                        segment_duration,
                        parameter,
                    )
                )

        dense_positions = np.asarray(dense_positions)
        dense_orientations = np.asarray(dense_orientations)
        dense_lengths = np.linalg.norm(
            np.diff(dense_positions, axis=0), axis=1
        ) + 0.06 * np.linalg.norm(
            np.diff(dense_orientations, axis=0), axis=1
        )
        cumulative_distance = np.concatenate(
            [[0.0], np.cumsum(dense_lengths)]
        )
        total_length = cumulative_distance[-1]
        timing_lengths = dense_lengths.copy()
        if approach_slowdown_distance > 0.0:
            segment_midpoints = (
                cumulative_distance[:-1] + cumulative_distance[1:]
            ) / 2
            remaining_distance = total_length - segment_midpoints
            slowdown_progress = np.clip(
                1.0 - remaining_distance / approach_slowdown_distance,
                0.0,
                1.0,
            )
            speed_scales = 1.0 - (
                1.0 - approach_speed_scale
            ) * np.array([smootherstep(value) for value in slowdown_progress])
            timing_lengths /= speed_scales
        cumulative_timing = np.concatenate(
            [[0.0], np.cumsum(timing_lengths)]
        )
        total_timing = cumulative_timing[-1]
        speed_variation = float(self.rng.uniform(0.97, 1.03))
        cruise_speed *= speed_variation
        motion_area = 1.0 - endpoint_ramp / 2 * (
            int(start_at_rest) + int(end_at_rest)
        )
        duration = total_timing / (cruise_speed * motion_area)
        steps = max(2, round(duration * FPS))

        for index in range(1, steps + 1):
            time_progress = index / steps
            distance_progress = motion_progress(
                time_progress,
                start_at_rest,
                end_at_rest,
                ramp=endpoint_ramp,
            )
            target_length = distance_progress * total_timing
            upper = min(
                int(np.searchsorted(cumulative_timing, target_length)),
                len(cumulative_timing) - 1,
            )
            lower = max(0, upper - 1)
            interval_length = cumulative_timing[upper] - cumulative_timing[lower]
            if interval_length <= 1e-9:
                mix = 0.0
            else:
                mix = (
                    target_length - cumulative_timing[lower]
                ) / interval_length
            position = (
                dense_positions[lower] * (1.0 - mix)
                + dense_positions[upper] * mix
            )
            orientation = (
                dense_orientations[lower] * (1.0 - mix)
                + dense_orientations[upper] * mix
            )
            if grip_keyframes is not None:
                self.grip[arm] = self.interpolated_grip(
                    time_progress, grip_keyframes
                )

            error = solver.solve(position, orientation)
            self.max_ik_error = max(self.max_ik_error, error)
            if attached_block is not None:
                self.block_orientations[attached_block] = (
                    orientation.copy()
                )
                self.append((arm, attached_block))
            else:
                self.append()

    def sampled_stack(
        self, center: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        layout = {}
        for layer, name in enumerate(("green", "yellow", "blue", "red")):
            xy = clipped_sample(
                self.rng,
                center,
                np.array([0.008, 0.008]),
                np.array([0.014, 0.014]),
            )
            position = np.array(
                [
                    xy[0],
                    xy[1],
                    BLOCK_HALF * (2 * layer + 1) + layer * 0.001,
                ]
            )
            orientation = clipped_sample(
                self.rng,
                np.zeros(3),
                np.array([0.020, 0.020, 0.120]),
                np.array([0.035, 0.035, 0.220]),
            )
            layout[name] = (position, orientation)
        return layout

    def pick_from_current(
        self,
        arm: str,
        block: str,
        gentle_start: bool,
    ) -> None:
        solver = self.solvers[arm]
        source = self.blocks[block].copy()
        source_orientation = self.block_orientations[block].copy()
        pregrasp, aligned = self.sampled_grasp_approach(source)
        states = [
            self.sampled_reach_bend(solver.position(), pregrasp),
            pregrasp,
            aligned,
        ]
        orientation_states = self.state_orientations(
            solver.current_orientation,
            source_orientation,
            fractions=(0.34, 0.76, 0.91),
            sigma=0.012,
            limit=0.022,
        )
        self.move_state_path(
            arm,
            states,
            source,
            orientation_states,
            source_orientation,
            cruise_speed=0.20 if gentle_start else 0.26,
            grip_keyframes=[
                (0.0, self.grip[arm]),
                (0.88, self.grip[arm]),
                (1.0, GRASP_AMOUNT),
            ],
            start_at_rest=gentle_start,
            approach_slowdown_distance=0.075,
            approach_speed_scale=0.72,
            endpoint_ramp=0.08,
        )

    def carry_and_place(
        self,
        arm: str,
        block: str,
        destination: np.ndarray,
        destination_orientation: np.ndarray,
    ) -> None:
        solver = self.solvers[arm]
        source = self.blocks[block].copy()
        source_orientation = self.block_orientations[block].copy()
        states = self.sampled_midair_positions(
            solver.position(),
            destination,
            start_lift=self.endpoint_clearance(source),
            end_drop=self.endpoint_clearance(destination),
        )
        orientation_states = self.state_orientations(
            source_orientation,
            destination_orientation,
            fractions=(0.10, 0.50, 0.90),
            sigma=0.018,
            limit=0.032,
        )
        self.move_state_path(
            arm,
            states,
            destination,
            orientation_states,
            destination_orientation,
            cruise_speed=0.27,
            attached_block=block,
            vertical_departure=True,
            vertical_approach=True,
        )

        self.blocks[block] = destination.copy()
        self.block_orientations[block] = destination_orientation.copy()

    def reposition_and_pick(
        self,
        arm: str,
        placed_destination: np.ndarray,
        next_block: str,
    ) -> None:
        solver = self.solvers[arm]
        source = self.blocks[next_block].copy()
        source_orientation = self.block_orientations[next_block].copy()
        pregrasp, aligned = self.sampled_grasp_approach(source)
        states = self.sampled_midair_positions(
            solver.position(),
            pregrasp,
            start_lift=self.endpoint_clearance(placed_destination),
            end_drop=0.0,
        )
        states.append(aligned)
        orientation_states = self.state_orientations(
            solver.current_orientation,
            source_orientation,
            fractions=(0.08, 0.42, 0.79, 0.92),
            sigma=0.020,
            limit=0.035,
        )
        self.move_state_path(
            arm,
            states,
            source,
            orientation_states,
            source_orientation,
            cruise_speed=0.265,
            grip_keyframes=[
                (0.0, GRASP_AMOUNT),
                (0.14, 0.0),
                (0.88, 0.0),
                (1.0, GRASP_AMOUNT),
            ],
            vertical_departure=True,
            approach_slowdown_distance=0.075,
            approach_speed_scale=0.72,
            endpoint_ramp=0.08,
        )

    def release_to_neutral(
        self,
        arm: str,
        placed_destination: np.ndarray,
        neutral: np.ndarray,
    ) -> None:
        solver = self.solvers[arm]
        start_clearance = self.endpoint_clearance(placed_destination)
        states = self.decision_positions(
            solver.position(),
            neutral,
            start_lift=start_clearance,
            air_lift=0.026,
            end_drop=-0.018,
            bend_sigma=0.024,
            bend_limit=0.042,
        )
        orientation_states = self.state_orientations(
            solver.current_orientation,
            np.zeros(3),
            sigma=0.016,
            limit=0.028,
        )
        self.move_state_path(
            arm,
            states,
            neutral,
            orientation_states,
            np.zeros(3),
            cruise_speed=0.24,
            grip_keyframes=[
                (0.0, GRASP_AMOUNT),
                (0.18, 0.0),
                (1.0, 0.0),
            ],
        )


def run_stack_sequence(
    timeline: Timeline,
    arm: str,
    order: tuple[str, ...],
    destination_layout: dict[str, tuple[np.ndarray, np.ndarray]],
    neutral: np.ndarray,
) -> None:
    timeline.pick_from_current(arm, order[0], gentle_start=True)
    for index, block in enumerate(order):
        destination, orientation = destination_layout[block]
        timeline.carry_and_place(
            arm,
            block,
            destination,
            orientation,
        )
        if index + 1 < len(order):
            timeline.reposition_and_pick(
                arm,
                destination,
                order[index + 1],
            )
        else:
            timeline.release_to_neutral(arm, destination, neutral)


def make_timeline(
    model: mujoco.MjModel, seed: int
) -> tuple[mujoco.MjData, Timeline]:
    data = mujoco.MjData(model)
    initial_q = np.array([-1.28, -1.43, 1.78, -1.91, -1.57, 0.0])
    data.qpos[:6] = initial_q
    data.qpos[14:20] = initial_q
    mujoco.mj_forward(model, data)

    timeline = Timeline(model, data, np.random.default_rng(seed))
    left_neutral = np.array([-0.27, 0.08, 0.43])
    right_neutral = np.array([0.27, 0.08, 0.43])
    neutral_orientation = np.zeros(3)
    timeline.max_ik_error = max(
        timeline.solvers["left"].solve(
            left_neutral, neutral_orientation, 500
        ),
        timeline.solvers["right"].solve(
            right_neutral, neutral_orientation, 500
        ),
    )
    loop_start_qpos = data.qpos.copy()
    timeline.hold(0.45)

    run_stack_sequence(
        timeline,
        "right",
        ("green", "yellow", "blue", "red"),
        timeline.sampled_stack(STACK_RIGHT),
        right_neutral,
    )
    timeline.hold(0.25)

    run_stack_sequence(
        timeline,
        "left",
        ("red", "blue", "yellow", "green"),
        LEFT_STACK_LAYOUT,
        left_neutral,
    )
    data.qpos[:] = loop_start_qpos
    timeline.solvers["left"].current_orientation = neutral_orientation.copy()
    timeline.solvers["right"].current_orientation = neutral_orientation.copy()
    timeline.hold(0.55)
    return data, timeline


def save_png(path: Path, pixels: np.ndarray) -> None:
    height, width, _ = pixels.shape
    raw = b"".join(b"\x00" + row.tobytes() for row in pixels)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    )
    png += chunk(b"IDAT", zlib.compress(raw, 9))
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def render_video(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    timeline: Timeline,
    video_path: Path,
    poster_path: Path | None,
    ffmpeg: str,
) -> None:
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{WIDTH}x{HEIGHT}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        "16",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(video_path),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [0.0, 0.03, 0.31]
    camera.distance = 1.82
    camera.azimuth = 90
    camera.elevation = -20

    body_mocap = {
        name: model.body_mocapid[
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, f"{name}_block"
            )
        ]
        for name in BLOCK_NAMES
    }

    try:
        with mujoco.Renderer(model, height=HEIGHT, width=WIDTH) as renderer:
            for frame_index, (qpos, blocks, block_orientations) in enumerate(
                timeline.frames
            ):
                data.qpos[:] = qpos
                for name in BLOCK_NAMES:
                    mocap_id = body_mocap[name]
                    data.mocap_pos[mocap_id] = blocks[name]
                    data.mocap_quat[mocap_id] = orientation_quaternion(
                        block_orientations[name]
                    )
                mujoco.mj_forward(model, data)
                renderer.update_scene(data, camera=camera)
                pixels = renderer.render()
                if frame_index == 0 and poster_path is not None:
                    poster_pixels = pixels.copy()
                    background = poster_pixels[0, 0].copy()
                    background_mask = np.all(
                        poster_pixels == background, axis=2
                    )
                    poster_pixels[background_mask] = [246, 242, 236]
                    save_png(poster_path, poster_pixels)
                if process.stdin is None:
                    raise RuntimeError("ffmpeg stdin closed unexpectedly")
                process.stdin.write(pixels.tobytes())
    finally:
        if process.stdin is not None:
            process.stdin.close()
        return_code = process.wait()
        if return_code:
            raise RuntimeError(f"ffmpeg exited with status {return_code}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--menagerie", type=Path, default=DEFAULT_MENAGERIE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[11, 29, 47, 73, 101]
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required to encode the website videos")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = build_model(args.menagerie)
    for index, seed in enumerate(args.seeds):
        data, timeline = make_timeline(model, seed)
        video_path = args.output_dir / f"block-stack-{seed}.mp4"
        poster_path = (
            args.output_dir / "block-stack-poster.png" if index == 0 else None
        )
        render_video(
            model,
            data,
            timeline,
            video_path,
            poster_path,
            ffmpeg,
        )
        duration = len(timeline.frames) / FPS
        print(
            f"seed={seed} duration={duration:.2f}s "
            f"max_ik_error={timeline.max_ik_error:.6f}m "
            f"unsafe_contacts={len(timeline.unsafe_contacts)} "
            f"bytes={video_path.stat().st_size}"
        )


if __name__ == "__main__":
    main()
