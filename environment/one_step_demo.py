from __future__ import annotations

import argparse
from pathlib import Path

from environment.one_step_planner import (
    OneStepPlanner,
    PlannerSettings,
    format_plan_result,
)
from environment.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="base端固定、搜索L8端最远可达候选点的单步规划演示"
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="仓库根目录，默认自动使用当前包所在仓库",
    )
    parser.add_argument(
        "--support-rank-from-bottom",
        type=int,
        default=1,
        help="支撑端从下往上第几个候选点，1表示最低点",
    )
    parser.add_argument(
        "--trajectory-steps",
        type=int,
        default=50,
        help="关节空间插值状态数量",
    )
    parser.add_argument(
        "--yaw-samples",
        type=int,
        default=16,
        help="目标吸盘绕法向的姿态采样数量",
    )
    parser.add_argument(
        "--attach-search-spacing-m",
        type=float,
        default=0.25,
        help="连续附着线粗搜索弧长间隔，单位m",
    )
    parser.add_argument(
        "--attach-refinement-spacing-m",
        type=float,
        default=0.125,
        help="最高可行区域局部细化的弧长间隔，单位m",
    )
    parser.add_argument(
        "--ik-random-seeds",
        type=int,
        default=3,
        help="每个目标额外使用的随机合法IK初值数量",
    )
    parser.add_argument(
        "--endpoint-only",
        action="store_true",
        help="只搜索最高合法终点，不验证整条轨迹（阶段2诊断）",
    )
    parser.add_argument(
        "--target-scan-limit",
        type=int,
        default=None,
        help="只检查距离最远的前N个目标点，默认检查全部",
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=50,
        help="规划时每检查多少个目标点打印一次进度；0表示不打印",
    )
    parser.add_argument(
        "--collision-margin-m",
        type=float,
        default=0.0,
        help="机器人与铁塔的安全距离阈值，单位m；0表示只检查实际接触/穿透",
    )
    parser.add_argument(
        "--allowed-contact-radius-m",
        type=float,
        default=0.09,
        help="吸盘中心附近允许接触半径，单位m",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="使用PyBullet GUI显示规划过程",
    )
    parser.add_argument(
        "--playback",
        action="store_true",
        help="规划成功后在GUI中播放轨迹",
    )
    parser.add_argument(
        "--playback-repeats",
        type=int,
        default=1,
        help="GUI播放轨迹重复次数",
    )
    parser.add_argument(
        "--playback-seconds-per-state",
        type=float,
        default=0.05,
        help="GUI播放时每个轨迹状态停留秒数",
    )
    parser.add_argument(
        "--keep-open",
        action="store_true",
        help="GUI模式下规划结束后保持窗口打开",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    settings = PlannerSettings(
        support_rank_from_bottom=args.support_rank_from_bottom,
        trajectory_steps=args.trajectory_steps,
        collision_margin_m=args.collision_margin_m,
        allowed_contact_radius_m=args.allowed_contact_radius_m,
        yaw_samples=args.yaw_samples,
        attach_search_spacing_m=args.attach_search_spacing_m,
        attach_refinement_spacing_m=args.attach_refinement_spacing_m,
        ik_random_seeds=args.ik_random_seeds,
        endpoint_only=args.endpoint_only,
        target_scan_limit=args.target_scan_limit,
        progress_interval=args.progress_interval,
    )

    # 先在DIRECT模式中规划出一条确定轨迹，再按需打开GUI播放。
    # 这样GUI里只展示最终轨迹，不展示目标搜索和IK试探过程。
    planner = OneStepPlanner(paths=paths, settings=settings, gui=False)
    if args.gui:
        print("先在DIRECT模式规划确定轨迹，成功后再打开GUI播放。", flush=True)
    result = planner.plan()
    print(format_plan_result(result))

    if args.gui:
        print("DIRECT规划阶段结束，准备打开GUI。", flush=True)
        planner.visualize_result(
            result,
            playback=args.playback,
            keep_open=args.keep_open,
            playback_repeats=args.playback_repeats,
            playback_seconds_per_state=args.playback_seconds_per_state,
        )


if __name__ == "__main__":
    main()
