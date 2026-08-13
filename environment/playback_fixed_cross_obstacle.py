"""只回放固定跨障实验已经保存的轨迹，不重新规划。"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from environment.fixed_cross_obstacle_demo import _play_best_effort, _play_saved
from environment.paths import ProjectPaths


def main() -> None:
    parser = argparse.ArgumentParser(description="回放固定跨障目标成功轨迹")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--keep-open", action="store_true", help="回放后保持GUI打开")
    parser.add_argument("--seconds-per-state", type=float, default=0.08, help="每个状态停留秒数")
    parser.add_argument("--best-effort", action="store_true", help="显示已保存的best-effort IK构型")
    parser.add_argument(
        "--animate-best-effort",
        action="store_true",
        help="播放Step 2终点到best-effort构型的诊断插值，不代表可行轨迹",
    )
    parser.add_argument(
        "--close-after",
        action="store_true",
        help="best-effort显示后立即关闭GUI；默认保持打开",
    )
    args = parser.parse_args()
    if "DISPLAY" not in os.environ:
        raise RuntimeError("启动GUI前请先执行：export DISPLAY=:1")
    paths = ProjectPaths.from_repo_root(args.repo_root)
    if args.best_effort:
        _play_best_effort(
            paths,
            # best-effort默认保持窗口，避免命令返回时with上下文自动断开GUI。
            keep_open=(args.keep_open or not args.close_after),
            animate=args.animate_best_effort,
            seconds_per_state=args.seconds_per_state,
        )
    else:
        _play_saved(paths, keep_open=args.keep_open, seconds_per_state=args.seconds_per_state)


if __name__ == "__main__":
    main()
