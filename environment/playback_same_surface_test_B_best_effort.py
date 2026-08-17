"""Test B最佳失败IK构型的GUI诊断回放入口。"""

from __future__ import annotations

from environment.playback_same_surface_best_effort import build_parser, run_from_args


def main() -> None:
    parser = build_parser("回放Test B：base_end在surface2上的最佳失败构型")
    run_from_args("B", parser.parse_args())


if __name__ == "__main__":
    main()
