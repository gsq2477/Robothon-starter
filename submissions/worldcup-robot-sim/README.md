# WorldCup Tactical Ball Drill

## Project name

WorldCup Tactical Ball Drill

## Robot platform

A MuJoCo planar robot that physically pushes a soccer ball into a target zone.

## Task goal

Use World Cup prediction rows as tactical inputs, then run a closed-loop MuJoCo ball drill. The robot reads live robot and ball positions, selects a tactic-specific target zone, and pushes the ball into that zone.

## Technical approach

`worldcup_sample.csv` provides reproducible prediction rows. Each row maps to a tactic:

- high-tempo press
- direct wide attack
- compact recovery
- balanced transition

The MuJoCo scene in `worldcup_ball_drill.xml` contains a field, robot, ball, walls, and target zone. `run_worldcup_robot_sim.py` uses a PD feedback controller over live MuJoCo joint and ball state. It runs 20 trials and records success rate, final error, control force, ball position, and robot position.

## Core features

- Closed-loop position feedback control.
- Real MuJoCo ball contact and target-zone task.
- 20-trial evaluation with success rate.
- Tactic selection from prediction CSV.
- Reproducible demo video and JSON trajectory log.

## Highlights

The previous version only visualized a prediction label. This version makes the prediction change a physical task parameter and measures whether the robot actually completes the drill in MuJoCo.

## Current limitations

The controller is a simple PD policy, not a learned soccer policy.

## Future improvements

Add multi-robot defenders, learned policies, and live prediction updates during the drill.

## How to run

From the repository root:

```bash
python -m pip install -r requirements.txt
python submissions/worldcup-robot-sim/run_worldcup_robot_sim.py --self-check
python submissions/worldcup-robot-sim/run_worldcup_robot_sim.py
```

## Demo video

Run the command above to generate:

- `submissions/worldcup-robot-sim/demo.mp4`
- `submissions/worldcup-robot-sim/worldcup_robot_plan.json`
- `submissions/worldcup-robot-sim/worldcup_robot_trajectory.json`
