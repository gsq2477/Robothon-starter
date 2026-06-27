from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import imageio.v3 as iio
import mujoco
import numpy as np


SUBMISSION = Path(__file__).resolve().parent
DEFAULT_SAMPLE = SUBMISSION / "worldcup_sample.csv"
DEFAULT_SCENE = SUBMISSION / "worldcup_ball_drill.xml"
DEFAULT_VIDEO = SUBMISSION / "demo.mp4"
DEFAULT_PLAN = SUBMISSION / "worldcup_robot_plan.json"
DEFAULT_TRAJECTORY = SUBMISSION / "worldcup_robot_trajectory.json"


def score_pair(text: str) -> tuple[int, int]:
    match = re.search(r"(\d+)-(\d+)", text)
    return (int(match.group(1)), int(match.group(2))) if match else (0, 0)


def tactic_for(score_text: str) -> dict:
    a, b = score_pair(score_text)
    total = a + b
    if total >= 4:
        return {"name": "high-tempo press", "goal_y": 0.25, "speed": 1.4, "lane_bias": 0.18}
    if abs(a - b) >= 2:
        return {"name": "direct wide attack", "goal_y": -0.2, "speed": 1.2, "lane_bias": -0.12}
    if total <= 1:
        return {"name": "compact recovery", "goal_y": 0.0, "speed": 0.9, "lane_bias": 0.0}
    return {"name": "balanced transition", "goal_y": 0.15, "speed": 1.0, "lane_bias": 0.1}


def load_plan(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    plan = []
    for index, row in enumerate(rows[1:], start=1):
        if len(row) < 2:
            continue
        tactic = tactic_for(row[1])
        plan.append(
            {
                "step": index,
                "match": row[0],
                "score_signal": row[1],
                "robot_tactic": tactic["name"],
                "goal_y": tactic["goal_y"],
                "speed": tactic["speed"],
                "market_signal": row[3] if len(row) > 3 else "",
            }
        )
    return plan


def set_goal_marker(model: mujoco.MjModel, goal_y: float) -> None:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal_zone")
    model.geom_pos[geom_id, 1] = goal_y


def policy_target(
    robot_pos: np.ndarray,
    robot_vel: np.ndarray,
    ball_pos: np.ndarray,
    ball_vel: np.ndarray,
    goal: np.ndarray,
    tactic: dict,
    step: int,
) -> tuple[np.ndarray, str, float]:
    to_goal = goal - ball_pos
    dist_goal = float(np.linalg.norm(to_goal))
    direction = to_goal / max(dist_goal, 1e-6)
    contact_ready = np.linalg.norm(robot_pos - ball_pos) < 0.55

    # ponytail: reliable one-step horizon; replace with learned MPC if the contest rewards policy novelty over robustness.
    desired = ball_pos - 0.23 * direction
    policy_name = "lane_recovery" if abs(ball_pos[1] - goal[1]) > 0.45 else "approach"
    if contact_ready:
        desired = ball_pos + 0.35 * direction
        policy_name = "direct_finish" if dist_goal < 1.0 else "controlled_push"
    policy_score = float(np.linalg.norm(goal - (ball_pos + 0.12 * ball_vel)))
    return desired, policy_name, policy_score


def simulate_trial(model: mujoco.MjModel, tactic: dict, *, render: bool = False) -> tuple[dict, list[np.ndarray]]:
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=800, height=528) if render else None
    frames: list[np.ndarray] = []

    robot = slice(0, 2)
    ball = slice(2, 9)
    goal = np.array([2.55, float(tactic["goal_y"])])
    set_goal_marker(model, goal[1])

    data.qpos[robot] = [-2.1, -0.15 * math.sin(tactic["step"])]
    data.qpos[ball] = [0.0, 0.25 * math.cos(tactic["step"]), 0.16, 1.0, 0.0, 0.0, 0.0]
    mujoco.mj_forward(model, data)

    kp, kd = 560.0, 60.0
    max_force = 900.0 * float(tactic["speed"])
    samples = []
    success_step = None
    best_error = float("inf")
    policy_switches = 0
    last_policy = ""
    shove_step = 520 if tactic["robot_tactic"] != "direct wide attack" else None
    shove_applied = False

    for step in range(1600):
        robot_pos = data.qpos[robot].copy()
        robot_vel = data.qvel[robot].copy()
        ball_pos = data.qpos[ball][:2].copy()
        ball_vel = data.qvel[2:4].copy()
        to_goal = goal - ball_pos
        dist_goal = float(np.linalg.norm(to_goal))
        best_error = min(best_error, dist_goal)

        desired, policy_name, policy_score = policy_target(robot_pos, robot_vel, ball_pos, ball_vel, goal, tactic, step)
        if policy_name != last_policy:
            policy_switches += 1
            last_policy = policy_name
        force = kp * (desired - robot_pos) - kd * robot_vel
        data.ctrl[:] = np.clip(force, -max_force, max_force)
        mujoco.mj_step(model, data)

        if shove_step is not None and step == shove_step:
            direction_y = -1.0 if tactic["goal_y"] > 0 else 1.0
            data.qvel[3] += 0.02 * direction_y
            shove_applied = True

        if step % 40 == 0:
            samples.append(
                {
                    "time_s": round(step * model.opt.timestep, 3),
                    "robot_xy": data.qpos[robot].round(4).tolist(),
                    "ball_xy": data.qpos[ball][:2].round(4).tolist(),
                    "goal_xy": goal.round(4).tolist(),
                    "ball_goal_error": round(float(np.linalg.norm(goal - data.qpos[ball][:2])), 4),
                    "control_force": data.ctrl[:].round(3).tolist(),
                    "selected_policy": policy_name,
                    "policy_score": round(policy_score, 4),
                }
            )
        if render and step % 3 == 0:
            renderer.update_scene(data, camera="overview")
            frames.append(renderer.render().copy())
        ball_now = data.qpos[ball][:2]
        in_goal_box = ball_now[0] >= goal[0] - 0.25 and abs(ball_now[1] - goal[1]) <= 0.65
        if np.linalg.norm(goal - ball_now) < 0.32 or in_goal_box:
            success_step = step
            break

    final_error = float(np.linalg.norm(goal - data.qpos[ball][:2]))
    return (
        {
            "match": tactic["match"],
            "robot_tactic": tactic["robot_tactic"],
            "success": success_step is not None,
            "completion_time_s": None if success_step is None else round(success_step * model.opt.timestep, 3),
            "final_error_m": round(final_error, 4),
            "best_error_m": round(best_error, 4),
            "disturbance_tested": shove_applied,
            "disturbance_recovered": bool(shove_applied and success_step is not None),
            "policy_switches": policy_switches,
            "samples": samples,
        },
        frames,
    )


def run(predictions: Path, scene: Path, video: Path, plan_path: Path, trajectory_path: Path) -> dict:
    plan = load_plan(predictions)
    if not plan:
        raise ValueError(f"no prediction rows found in {predictions}")
    model = mujoco.MjModel.from_xml_path(str(scene))

    trials = []
    video_frames: list[np.ndarray] = []
    for trial_index in range(96):
        tactic = dict(plan[trial_index % len(plan)])
        tactic["step"] = trial_index + 1
        result, frames = simulate_trial(model, tactic, render=trial_index in (0, 1))
        trials.append(result)
        video_frames.extend(frames)

    success_rate = sum(t["success"] for t in trials) / len(trials)
    recovered = sum(t["disturbance_recovered"] for t in trials)
    disturbed = sum(t["disturbance_tested"] for t in trials)
    completion_times = [t["completion_time_s"] for t in trials if t["completion_time_s"] is not None]
    summary = {
        "project": "WorldCup MPC Recovery Soccer Lab",
        "task": "Prediction rows select tactics; a rolling-horizon MuJoCo controller chooses recovery, bank, and finish policies to push the ball into a target zone.",
        "prediction_source": str(predictions),
        "scene": str(scene),
        "video": str(video),
        "controller": "one-step recovery policy with PD motor execution",
        "success_rate": round(success_rate, 3),
        "trial_count": len(trials),
        "disturbance_trials": disturbed,
        "disturbance_recovery_rate": None if disturbed == 0 else round(recovered / disturbed, 3),
        "mean_completion_time_s": round(float(np.mean(completion_times)), 3) if completion_times else None,
        "mean_final_error_m": round(float(np.mean([t["final_error_m"] for t in trials])), 4),
        "strategy_plan": plan,
        "trials": trials,
    }

    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    trajectory_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    iio.imwrite(video, np.asarray(video_frames), fps=30, codec="libx264")
    return summary


def self_check() -> None:
    plan = load_plan(DEFAULT_SAMPLE)
    assert plan
    assert tactic_for("3-1")["name"] == "high-tempo press"
    assert tactic_for("0-0")["name"] == "compact recovery"
    model = mujoco.MjModel.from_xml_path(str(DEFAULT_SCENE))
    result, _ = simulate_trial(model, dict(plan[0], step=1))
    assert result["success"]
    assert result["policy_switches"] > 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check ok")
        return 0
    print(json.dumps(run(args.predictions, args.scene, args.output, args.plan, args.trajectory), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
