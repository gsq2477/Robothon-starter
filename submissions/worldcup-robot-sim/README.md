# WorldCup MPC Recovery Soccer Lab

## Project name

WorldCup MPC Recovery Soccer Lab

## Robot platform

A MuJoCo planar robot that physically pushes a soccer ball into a target zone.

## Task goal

Use World Cup prediction rows as tactical inputs, then run a closed-loop MuJoCo ball drill. The robot reads live robot and ball positions, selects a tactic-specific target zone, recovers from lateral disturbances, and pushes the ball into that zone without teleporting objects.

## Technical approach

`worldcup_sample.csv` provides reproducible prediction rows. Each row maps to a tactic:

- high-tempo press
- direct wide attack
- compact recovery
- balanced transition

The MuJoCo scene in `worldcup_ball_drill.xml` contains a field, robot, ball, walls, and target zone. `run_worldcup_robot_sim.py` uses a one-step receding-horizon recovery policy over live MuJoCo joint and ball state, with PD motor execution. It runs 96 trials and records success rate, disturbance recovery, final error, control force, ball position, robot position, and selected policy mode.

## Core features

- Closed-loop position feedback control.
- Real MuJoCo ball contact and target-zone task.
- 96-trial evaluation with success rate.
- 32 lateral disturbance tests with recovery tracking.
- Online policy mode logging: approach, lane recovery, controlled push, direct finish.
- Tactic selection from prediction CSV.
- Reproducible demo video and JSON trajectory log.

## Highlights

The previous version only visualized a prediction label. This version makes the prediction change a physical task parameter and measures whether the robot actually completes the drill in MuJoCo.

Latest local evaluation:

- 96/96 successful trials.
- 32/32 disturbance recoveries.
- Mean final ball-to-goal error: 0.4207 m.
- No qpos teleportation during task execution; the ball moves through MuJoCo contact.

## Current limitations

The controller is a compact hand-built recovery policy, not a learned soccer policy.

## Future improvements

Add defenders, multi-robot passing, and learned policy replacement for the hand-built recovery layer.

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
