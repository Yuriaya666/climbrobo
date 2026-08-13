"""独立的foot2位置工作空间诊断。

本文件故意不调用OneStepPlanner，不做法向、碰撞、Straight或RRT。
唯一优化量是8个实际关节角，唯一目标函数是foot2/l8_end吸盘中心的
世界位置与固定target_xyz之间的欧氏距离。
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import differential_evolution, least_squares
from scipy.stats import qmc

from environment.paths import ProjectPaths
from environment.rebase import RebaseResult, rebase_to_support
from environment.scene import PyBulletScene
from environment.suction_frames import SuctionFrameSet
from environment.trajectory_io import SavedTrajectory
from environment.transforms import RigidTransform


TARGET_XYZ_M = np.array([0.06999983, 0.0, 1.12199991], dtype=float)


@dataclass(frozen=True)
class WorkspaceState:
    """固定支撑端后的工作空间测试状态。"""

    q_start: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    joint_names: tuple[str, ...]
    support_pose: RigidTransform
    base_pose: RigidTransform
    rebase: RebaseResult


@dataclass(frozen=True)
class SearchResult:
    """一次求解器运行的最佳位置构型。"""

    method: str
    seed: int
    q: np.ndarray
    xyz: np.ndarray
    error_m: float
    iterations: int
    evaluations: int
    runtime_s: float


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="固定foot1后的foot2 position-only工作空间测试")
    parser.add_argument("--repo-root", type=Path, default=None, help="仓库根目录")
    parser.add_argument("--pybullet-seeds", type=int, default=32, help="PyBullet position-only IK初值数量")
    parser.add_argument("--de-seeds", type=int, default=10, help="Differential Evolution独立随机种子数量")
    parser.add_argument("--de-maxiter", type=int, default=120, help="每个DE seed的最大迭代数")
    parser.add_argument("--de-popsize", type=int, default=10, help="每个DE seed的人口倍数")
    parser.add_argument("--polish-max-nfev", type=int, default=2500, help="局部least-squares最大函数评估数")
    parser.add_argument("--sobol-power", type=int, default=12, help="Sobol采样数量为2的该次幂")
    parser.add_argument("--gui", action="store_true", help="显示最终误差最小构型")
    parser.add_argument("--keep-open", action="store_true", help="GUI显示后保持打开")
    return parser


def _restore_step2_and_rebase(paths: ProjectPaths, frames: SuctionFrameSet) -> WorkspaceState:
    """读取Step 2末帧，并把foot1/base_end变为严格固定运动学根端。"""

    step2 = SavedTrajectory.load(paths.successful_step2_trajectory_npz)
    if step2.base_position_m is None or step2.base_orientation_xyzw is None:
        raise ValueError("第二步轨迹缺少base pose序列，无法严格恢复末帧")

    q_start = np.asarray(step2.trajectory_rad[-1], dtype=float)
    base_pose = RigidTransform(
        position=step2.base_position_m[-1],
        quaternion_xyzw=step2.base_orientation_xyzw[-1],
    )
    with PyBulletScene(paths, gui=False) as scene:
        scene.load_robot(base_pose)
        scene.reset_joints(q_start)
        support_before = scene.get_suction_pose(frames.base_end)
        rebase = rebase_to_support(
            scene,
            joints=q_start,
            old_support_name="foot2/l8_end",
            new_support_name="foot1/base_end",
            new_support_frame=frames.base_end,
            target_support_pose=support_before,
        )
        if not rebase.continuous:
            raise RuntimeError("rebase连续性检查失败，停止position-only工作空间测试")
        lower = scene.joint_lower_limits()
        upper = scene.joint_upper_limits()
        joint_names = tuple(joint.name for joint in scene.joints)

    if len(q_start) != 8 or len(lower) != 8:
        raise ValueError(
            f"本测试要求8个实际可动关节，实际q={len(q_start)}、limits={len(lower)}"
        )
    if np.any(upper <= lower):
        raise ValueError("URDF关节上下限无效")

    print("========== Position-only workspace test ==========")
    print("support: foot1/base_end（严格固定运动学根端）")
    print("moving: foot2/l8_end")
    print(f"target xyz: {TARGET_XYZ_M.tolist()} m")
    print(f"joint count: {len(q_start)}")
    print(f"joint names: {list(joint_names)}")
    for name, low, high in zip(joint_names, lower, upper):
        print(f"  {name}: [{low:.9f}, {high:.9f}] rad")
    print(
        "rebase continuity: "
        f"max position={rebase.max_link_position_change_m:.3e} m, "
        f"max orientation={rebase.max_link_orientation_change_rad:.3e} rad"
    )
    return WorkspaceState(
        q_start=q_start,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        support_pose=rebase.target_support_pose,
        base_pose=rebase.base_pose,
        rebase=rebase,
    )


class PositionOnlyFK:
    """只提供固定支撑锚定下的foot2吸盘中心位置函数。"""

    def __init__(self, paths: ProjectPaths, state: WorkspaceState, *, gui: bool = False) -> None:
        self.scene = PyBulletScene(paths, gui=gui)
        self.state = state
        self.frames = SuctionFrameSet.load(paths.suction_config)

    def __enter__(self) -> "PositionOnlyFK":
        self.scene.__enter__()
        self.scene.load_robot(self.state.base_pose)
        self.scene.enable_support_anchor(self.frames.base_end, self.state.support_pose)
        self.scene.reset_joints(self.state.q_start)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.scene.__exit__(exc_type, exc, traceback)

    def evaluate(self, q: np.ndarray) -> np.ndarray:
        """计算实际foot2/l8_end吸盘中心世界XYZ，不读取法向或姿态。"""

        value = np.asarray(q, dtype=float)
        if value.shape != (8,):
            raise ValueError(f"q必须是(8,)，实际为{value.shape}")
        self.scene.reset_joints(value)
        return self.scene.get_suction_pose(self.frames.l8_end).position.copy()

    def error(self, q: np.ndarray) -> float:
        return float(np.linalg.norm(self.evaluate(q) - TARGET_XYZ_M))


def _pybullet_position_only(
    fk: PositionOnlyFK,
    state: WorkspaceState,
    *,
    seed_count: int,
    max_iterations: int = 1000,
) -> SearchResult:
    """使用PyBullet原生position-only IK，不传targetOrientation。"""

    rng = np.random.default_rng(20260813)
    seeds = [state.q_start.copy()]
    seeds.extend(rng.uniform(state.lower, state.upper) for _ in range(max(0, seed_count - 1)))
    link_index = fk.scene.link_index(fk.frames.l8_end.link_name)
    ranges = state.upper - state.lower
    best: SearchResult | None = None
    started = time.perf_counter()

    for seed_index, rest_pose in enumerate(seeds):
        # 这里故意不传targetOrientation。PyBullet原生接口的目标是
        # L8 link的position；返回结果随后统一用实际l8_end吸盘中心FK评估。
        solution = p.calculateInverseKinematics(
            bodyUniqueId=fk.scene.robot_id,
            endEffectorLinkIndex=link_index,
            targetPosition=TARGET_XYZ_M.tolist(),
            lowerLimits=state.lower.tolist(),
            upperLimits=state.upper.tolist(),
            jointRanges=ranges.tolist(),
            restPoses=np.asarray(rest_pose, dtype=float).tolist(),
            maxNumIterations=max_iterations,
            residualThreshold=1e-8,
        )
        q = fk.scene.normalize_revolute_solutions(np.asarray(solution[:8], dtype=float))
        q = np.clip(q, state.lower, state.upper)
        xyz = fk.evaluate(q)
        error_m = float(np.linalg.norm(xyz - TARGET_XYZ_M))
        candidate = SearchResult(
            method="PYBULLET_POSITION_ONLY",
            seed=seed_index,
            q=q.copy(),
            xyz=xyz.copy(),
            error_m=error_m,
            iterations=max_iterations,
            evaluations=1,
            runtime_s=time.perf_counter() - started,
        )
        if best is None or candidate.error_m < best.error_m:
            best = candidate

    if best is None:
        raise RuntimeError("PyBullet position-only IK没有产生结果")
    return best


def _differential_evolution_runs(
    fk: PositionOnlyFK,
    state: WorkspaceState,
    *,
    seed_count: int,
    maxiter: int,
    popsize: int,
) -> list[SearchResult]:
    """运行多个独立DE，目标函数严格只有位置误差。"""

    bounds = list(zip(state.lower.tolist(), state.upper.tolist()))
    results: list[SearchResult] = []
    for seed in range(seed_count):
        started = time.perf_counter()
        result = differential_evolution(
            lambda q: fk.error(q),
            bounds=bounds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-9,
            atol=1e-10,
            polish=False,
            updating="immediate",
            workers=1,
        )
        q = np.clip(np.asarray(result.x, dtype=float), state.lower, state.upper)
        xyz = fk.evaluate(q)
        results.append(
            SearchResult(
                method="DIFFERENTIAL_EVOLUTION",
                seed=seed,
                q=q.copy(),
                xyz=xyz.copy(),
                error_m=float(np.linalg.norm(xyz - TARGET_XYZ_M)),
                iterations=int(result.nit),
                evaluations=int(result.nfev),
                runtime_s=time.perf_counter() - started,
            )
        )
        print(
            f"DE seed={seed}: best error={results[-1].error_m:.9f} m, "
            f"evals={result.nfev}, time={results[-1].runtime_s:.2f} s",
            flush=True,
        )
    return results


def _local_polish(fk: PositionOnlyFK, state: WorkspaceState, q_initial: np.ndarray, max_nfev: int) -> SearchResult:
    """以DE全局最优为初值，用有界least-squares只精修XYZ残差。"""

    started = time.perf_counter()
    result = least_squares(
        lambda q: fk.evaluate(q) - TARGET_XYZ_M,
        x0=np.clip(np.asarray(q_initial, dtype=float), state.lower, state.upper),
        bounds=(state.lower, state.upper),
        max_nfev=max_nfev,
        diff_step=1e-5,
        xtol=1e-12,
        ftol=1e-12,
        gtol=1e-12,
        verbose=0,
    )
    q = np.clip(np.asarray(result.x, dtype=float), state.lower, state.upper)
    xyz = fk.evaluate(q)
    return SearchResult(
        method="LOCAL_LEAST_SQUARES",
        seed=-1,
        q=q.copy(),
        xyz=xyz.copy(),
        error_m=float(np.linalg.norm(xyz - TARGET_XYZ_M)),
        iterations=int(result.nfev),
        evaluations=int(result.nfev),
        runtime_s=time.perf_counter() - started,
    )


def _sobol_sanity(fk: PositionOnlyFK, state: WorkspaceState, power: int) -> SearchResult:
    """用Sobol均匀采样做非主求解器的工作空间sanity check。"""

    count = 2 ** max(1, int(power))
    sampler = qmc.Sobol(d=8, scramble=True, seed=20260813)
    unit_samples = sampler.random_base2(m=max(1, int(power)))
    best_q = None
    best_xyz = None
    best_error = float("inf")
    started = time.perf_counter()
    for unit in unit_samples:
        q = state.lower + unit * (state.upper - state.lower)
        xyz = fk.evaluate(q)
        error_m = float(np.linalg.norm(xyz - TARGET_XYZ_M))
        if error_m < best_error:
            best_q = q.copy()
            best_xyz = xyz.copy()
            best_error = error_m
    if best_q is None or best_xyz is None:
        raise RuntimeError("Sobol没有产生样本")
    return SearchResult(
        method="SOBOL_SANITY",
        seed=20260813,
        q=best_q,
        xyz=best_xyz,
        error_m=best_error,
        iterations=count,
        evaluations=count,
        runtime_s=time.perf_counter() - started,
    )


def _write_results(path: Path, results: list[SearchResult], joint_names: tuple[str, ...]) -> None:
    """保存所有方法和seed的误差、位置、关节角及预算统计。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "method", "seed", "error_m", "x_m", "y_m", "z_m",
        "iterations", "evaluations", "runtime_s", *joint_names,
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            row = {
                "method": result.method,
                "seed": result.seed,
                "error_m": result.error_m,
                "x_m": result.xyz[0],
                "y_m": result.xyz[1],
                "z_m": result.xyz[2],
                "iterations": result.iterations,
                "evaluations": result.evaluations,
                "runtime_s": result.runtime_s,
            }
            row.update(dict(zip(joint_names, result.q.tolist())))
            writer.writerow(row)


def _save_summary(path: Path, state: WorkspaceState, results: list[SearchResult], best: SearchResult) -> None:
    """保存便于后续GUI或脚本复核的汇总数组。"""

    de_results = [item for item in results if item.method == "DIFFERENTIAL_EVOLUTION"]
    pybullet = next(item for item in results if item.method == "PYBULLET_POSITION_ONLY")
    local = next(item for item in results if item.method == "LOCAL_LEAST_SQUARES")
    sobol = next(item for item in results if item.method == "SOBOL_SANITY")
    np.savez(
        path,
        target_xyz_m=TARGET_XYZ_M,
        pybullet_best_q=pybullet.q,
        pybullet_best_xyz=pybullet.xyz,
        pybullet_best_error_m=np.asarray(pybullet.error_m),
        de_seed=np.asarray([item.seed for item in de_results], dtype=np.int64),
        de_error_m=np.asarray([item.error_m for item in de_results]),
        de_q=np.asarray([item.q for item in de_results]),
        de_xyz=np.asarray([item.xyz for item in de_results]),
        global_best_q=best.q,
        global_best_xyz=best.xyz,
        global_best_error_m=np.asarray(best.error_m),
        polished_q=local.q,
        polished_xyz=local.xyz,
        polished_error_m=np.asarray(local.error_m),
        sobol_best_q=sobol.q,
        sobol_best_xyz=sobol.xyz,
        sobol_best_error_m=np.asarray(sobol.error_m),
        lower_limits=state.lower,
        upper_limits=state.upper,
        joint_names=np.asarray(state.joint_names),
        unit=np.asarray("m/rad"),
    )


def _draw_result(paths: ProjectPaths, state: WorkspaceState, result: SearchResult, *, keep_open: bool) -> None:
    """GUI只显示最终位置误差最小构型、目标点、实际点和误差线。"""

    frames = SuctionFrameSet.load(paths.suction_config)
    with PyBulletScene(paths, gui=True) as scene:
        scene.load_tower()
        scene.load_robot(state.base_pose)
        scene.enable_support_anchor(frames.base_end, state.support_pose)
        scene.reset_joints(state.q_start)
        scene.highlight_robot()
        scene.draw_point(TARGET_XYZ_M, [1.0, 0.85, 0.0], size=16.0)
        # result.xyz是最终优化构型下的实际foot2吸盘中心，而不是起始状态。
        scene.draw_point(result.xyz, [0.1, 1.0, 0.1], size=14.0)
        scene.draw_polyline(np.vstack((TARGET_XYZ_M, result.xyz)), [1.0, 0.1, 0.1], radius=0.002)
        scene.draw_point(state.support_pose.position, [1.0, 0.1, 0.1], size=14.0)
        scene.focus_camera_on_robot(distance=1.5, yaw=45.0, pitch=-20.0)
        scene.reset_joints(result.q)
        print(
            f"GUI显示最终构型：actual={result.xyz.tolist()}，error={result.error_m:.9f} m",
            flush=True,
        )
        if keep_open:
            print("GUI保持打开，按Ctrl+C退出。", flush=True)
            try:
                while p.isConnected():
                    scene.reset_joints(result.q)
                    time.sleep(1.0 / 60.0)
            except KeyboardInterrupt:
                pass


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    paths.validate_required_files()
    frames = SuctionFrameSet.load(paths.suction_config)
    state = _restore_step2_and_rebase(paths, frames)

    with PositionOnlyFK(paths, state, gui=False) as fk:
        pybullet_result = _pybullet_position_only(
            fk,
            state,
            seed_count=max(1, int(args.pybullet_seeds)),
        )
        print(
            f"PyBullet position-only best: error={pybullet_result.error_m:.9f} m, "
            f"xyz={pybullet_result.xyz.tolist()}",
            flush=True,
        )
        de_results = _differential_evolution_runs(
            fk,
            state,
            seed_count=max(10, int(args.de_seeds)),
            maxiter=max(1, int(args.de_maxiter)),
            popsize=max(4, int(args.de_popsize)),
        )
        global_de = min(de_results, key=lambda item: item.error_m)
        local_result = _local_polish(
            fk,
            state,
            global_de.q,
            max_nfev=max(100, int(args.polish_max_nfev)),
        )
        sobol_result = _sobol_sanity(fk, state, max(1, int(args.sobol_power)))
        all_results = [pybullet_result, *de_results, local_result, sobol_result]
        final_result = min(all_results, key=lambda item: item.error_m)
        _write_results(paths.position_workspace_test_csv, all_results, state.joint_names)
        _save_summary(paths.position_workspace_test_npz, state, all_results, final_result)

    print("\n========== Position-only result ==========")
    print(f"Target xyz: {TARGET_XYZ_M.tolist()} m")
    print("PyBullet position-only IK:")
    print(f"  best error: {pybullet_result.error_m:.9f} m")
    print(f"  achieved xyz: {pybullet_result.xyz.tolist()}")
    print("Differential Evolution:")
    for result in de_results:
        print(f"  seed {result.seed}: best error={result.error_m:.9f} m")
    print(f"  global best error: {global_de.error_m:.9f} m")
    print(f"  achieved xyz: {global_de.xyz.tolist()}")
    print("Local polishing:")
    print(f"  final best error: {local_result.error_m:.9f} m")
    print(f"  achieved xyz: {local_result.xyz.tolist()}")
    print("Random/Sobol sanity check:")
    print(f"  best error: {sobol_result.error_m:.9f} m")
    print(f"  achieved xyz: {sobol_result.xyz.tolist()}")
    print(f"最终全方法最小误差: {final_result.error_m:.9f} m ({final_result.method})")
    if final_result.error_m < 0.005:
        print("最终判断: 目标XYZ在当前关节限位下的位置工作空间内。")
    elif final_result.error_m <= 0.020:
        print("最终判断: 边界区域，暂不判断可达或不可达。")
    else:
        print("最终判断: 多种位置-only方法均未达到5 mm，目标很可能在工作空间之外。")
    print(f"逐seed结果CSV: {paths.position_workspace_test_csv}（位置工作空间诊断）")
    print(f"汇总NPZ: {paths.position_workspace_test_npz}（位置工作空间诊断汇总）")

    if args.gui:
        _draw_result(paths, state, final_result, keep_open=args.keep_open)


if __name__ == "__main__":
    main()
