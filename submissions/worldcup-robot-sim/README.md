# WorldCup Robot Simulation

## Project name

WorldCup Robot Simulation

## Robot platform

FF Master humanoid model from `assets/Master/scene.xml`.

## Task goal

Use the local World Cup prediction model output as a tactical signal source, then generate a MuJoCo humanoid demo and trajectory plan from those rows.

## Technical approach

The submission keeps the prediction model outside this repository and reads its latest CSV output when available:

`F:\世界杯预测模型\results\2026-06-25\每日推荐表.csv`

If that local file is not present, the script uses the included `worldcup_sample.csv` so judges can still reproduce the MuJoCo demo.

## Core features

- Converts match score signals into robot tactics.
- Runs the official FF Master MuJoCo model.
- Produces `demo.mp4`, `worldcup_robot_plan.json`, and `worldcup_robot_trajectory.json`.
- Includes a bundled sample CSV for reproducible judging.

## Highlights

This is a lightweight bridge from an existing sports prediction pipeline to an embodied simulation task: prediction rows become tactical movement labels, then the humanoid demo records the resulting plan beside a MuJoCo-rendered video.

## Current limitations

The robot motion is a deterministic visualization, not a closed-loop soccer controller.

## Future improvements

Map each tactic to distinct joint trajectories, add a ball and goal scene, and connect live prediction updates to a multi-agent robot drill.

## How to run

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/worldcup-robot-sim/run_worldcup_robot_sim.py --self-check
python submissions/worldcup-robot-sim/run_worldcup_robot_sim.py
```

To force a specific prediction CSV:

```bash
python submissions/worldcup-robot-sim/run_worldcup_robot_sim.py --predictions path/to/predictions.csv
```

## Demo video

Run the command above to generate:

- `submissions/worldcup-robot-sim/demo.mp4`
- `submissions/worldcup-robot-sim/worldcup_robot_plan.json`
- `submissions/worldcup-robot-sim/worldcup_robot_trajectory.json`
