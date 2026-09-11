# Block-stacking simulation

This script renders the randomized episodes used by the website. It uses the
official UR5e and Robotiq 2F-85 models from
[MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).

Each seed samples small offsets around the grasp, lift, transition, placement,
and retreat targets. Damped least-squares inverse kinematics solves the arm
poses, and cubic Hermite splines carry velocity through intermediate targets.
The curves are resampled by arc length to keep Cartesian speed nearly constant,
with short acceleration ramps only at physical grasp and placement endpoints.
The fingers close during the final approach and open during retreat rather
than inserting stationary grasp poses. The destination stack also gets a few
millimeters of position variation plus sampled roll, pitch, and yaw per block.

## Generate the videos

Install `ffmpeg`, clone MuJoCo Menagerie, and create a Python environment:

```sh
python3 -m venv .venv
.venv/bin/pip install -r tools/simulation/requirements.txt
git clone --depth 1 https://github.com/google-deepmind/mujoco_menagerie.git
```

Then render the default five episodes:

```sh
MUJOCO_MENAGERIE=/path/to/mujoco_menagerie \
  .venv/bin/python tools/simulation/generate_block_stack.py
```

Pass different seeds to grow or replace the episode bank:

```sh
.venv/bin/python tools/simulation/generate_block_stack.py --seeds 5 8 13
```

The videos and poster are written to `assets/simulation/` by default.
