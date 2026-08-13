"""只回放已经保存的第一步和第二步，不重新规划。"""

from __future__ import annotations

import argparse
from pathlib import Path

from environment.paths import ProjectPaths
from environment.trajectory_io import SavedTrajectory
from environment.two_step_demo import _playback_two_steps


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="播放已保存的两步连续轨迹")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--seconds-per-state", type=float, default=0.08, help="每个状态停留秒数")
    parser.add_argument("--keep-open", action="store_true", help="播放结束后保持GUI打开")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    step1 = SavedTrajectory.load(paths.successful_trajectory_npz)
    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    print(
        f"读取两步轨迹：{paths.successful_two_step_trajectory_npz}（两步拼接轨迹数据）",
        flush=True,
    )
    print("启动GUI前使用 DISPLAY=:1。", flush=True)
    _playback_two_steps(
        paths,
        step1,
        step2,
        keep_open=args.keep_open,
        seconds_per_state=args.seconds_per_state,
    )


if __name__ == "__main__":
    main()
