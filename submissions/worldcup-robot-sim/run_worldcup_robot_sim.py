from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = Path(__file__).resolve().parent
DEFAULT_SAMPLE = SUBMISSION / "worldcup_sample.csv"
DEFAULT_LOCAL_MODEL_OUTPUT = Path(r"F:\世界杯预测模型\results\2026-06-25\每日推荐表.csv")
DEFAULT_VIDEO = SUBMISSION / "demo.mp4"
DEFAULT_TRAJECTORY = SUBMISSION / "worldcup_robot_trajectory.json"
DEFAULT_PLAN = SUBMISSION / "worldcup_robot_plan.json"


def score_pair(text: str) -> tuple[int, int]:
    match = re.search(r"(\d+)-(\d+)", text)
    if not match:
        return 0, 0
    return int(match.group(1)), int(match.group(2))


def tactic_for(score_text: str) -> str:
    a, b = score_pair(score_text)
    if a + b >= 4:
        return "high-tempo attack and recovery runs"
    if abs(a - b) >= 2:
        return "wide overload toward the favored side"
    if a + b <= 1:
        return "compact defensive shape"
    return "balanced press with controlled transition"


def load_plan(csv_path: Path) -> list[dict]:
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    plan = []
    for index, row in enumerate(rows[1:], start=1):
        if len(row) < 2:
            continue
        score = row[1]
        plan.append(
            {
                "step": index,
                "match": row[0],
                "score_signal": score,
                "robot_tactic": tactic_for(score),
                "market_signal": row[3] if len(row) > 3 else "",
            }
        )
    return plan


def run(predictions: Path, video: Path, trajectory: Path, plan_path: Path) -> dict:
    sys.path.insert(0, str(ROOT))
    from examples.run_ff_master_demo import DEFAULT_MODEL, run_demo

    plan = load_plan(predictions)
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = run_demo(
        model_path=DEFAULT_MODEL,
        video_path=video,
        trajectory_path=trajectory,
        duration_s=max(4.0, min(9.0, 2.0 + len(plan))),
        fps=24,
        width=640,
        height=480,
    )
    summary["project"] = "WorldCup Robot Simulation"
    summary["task"] = "Convert World Cup prediction rows into a MuJoCo humanoid strategy demo."
    summary["prediction_source"] = str(predictions)
    summary["strategy_plan"] = plan
    trajectory.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def self_check() -> None:
    assert score_pair("main 2-0; safe 1-0") == (2, 0)
    assert tactic_for("3-1") == "high-tempo attack and recovery runs"
    assert tactic_for("0-0") == "compact defensive shape"
    assert load_plan(DEFAULT_SAMPLE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--self-check", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.self_check:
        self_check()
        print("self-check ok")
        return 0

    predictions = args.predictions
    if predictions is None:
        predictions = DEFAULT_LOCAL_MODEL_OUTPUT if DEFAULT_LOCAL_MODEL_OUTPUT.exists() else DEFAULT_SAMPLE
    summary = run(predictions, args.output, args.trajectory, args.plan)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
