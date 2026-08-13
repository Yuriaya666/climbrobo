from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProjectPaths:
    """集中管理仓库内会被规划器读取的文件路径。"""

    repo_root: Path

    @classmethod
    def from_repo_root(cls, repo_root: Path | None = None) -> "ProjectPaths":
        if repo_root is None:
            repo_root = Path(__file__).resolve().parents[1]
        return cls(repo_root=repo_root.resolve())

    @property
    def robot_urdf(self) -> Path:
        return (
            self.repo_root
            / "robots_model"
            / "urdf"
            / "00样机（更换电机）-x12.SLDASM.urdf"
        )

    @property
    def robot_mesh_dir(self) -> Path:
        return self.repo_root / "robots_model" / "meshes"

    @property
    def suction_config(self) -> Path:
        return self.repo_root / "robots_model" / "config" / "suction_frames.yaml"

    @property
    def tower_collision_mesh(self) -> Path:
        return self.repo_root / "models" / "Tower.STL"

    @property
    def tower_visual_mesh(self) -> Path:
        return self.repo_root / "models" / "Tower_visual.STL"

    @property
    def candidate_dir(self) -> Path:
        return self.repo_root / "models" / "candidate_output"

    def candidate_npz(self, foot_name: str) -> Path:
        return self.candidate_dir / f"{foot_name}_candidates.npz"

    def attach_lines_npz(self, foot_name: str) -> Path:
        """连续附着线NPZ路径，不影响旧候选点文件。"""

        return self.candidate_dir / f"{foot_name}_attach_lines.npz"

    @property
    def successful_trajectory_npz(self) -> Path:
        """最近一次成功单步轨迹的NPZ数据文件。"""

        return self.candidate_dir / "successful_one_step_trajectory.npz"

    @property
    def successful_trajectory_csv(self) -> Path:
        """最近一次成功单步轨迹的CSV表格文件。"""

        return self.candidate_dir / "successful_one_step_trajectory.csv"

    @property
    def successful_step2_trajectory_npz(self) -> Path:
        """第二步成功关节轨迹的NPZ数据文件。"""

        return self.candidate_dir / "successful_step2_trajectory.npz"

    @property
    def successful_step2_trajectory_csv(self) -> Path:
        """第二步逐状态关节角的CSV表格文件。"""

        return self.candidate_dir / "successful_step2_trajectory.csv"

    @property
    def successful_two_step_trajectory_npz(self) -> Path:
        """第一步和第二步拼接轨迹的NPZ数据文件。"""

        return self.candidate_dir / "successful_two_step_trajectory.npz"

    @property
    def step3_higher_goal_diagnostics_csv(self) -> Path:
        """第三步高于当前高度的终点诊断CSV表格。"""

        return self.candidate_dir / "step3_higher_goal_diagnostics.csv"

    @property
    def step3_higher_goal_candidates_npz(self) -> Path:
        """第三步通过终点检查的高处q_goal候选数据。"""

        return self.candidate_dir / "step3_higher_goal_candidates.npz"

    @property
    def step3_trajectory_npz(self) -> Path:
        """第三步成功关节轨迹数据文件。"""

        return self.candidate_dir / "successful_step3_trajectory.npz"

    @property
    def step3_trajectory_csv(self) -> Path:
        """第三步成功逐状态关节角CSV表格。"""

        return self.candidate_dir / "successful_step3_trajectory.csv"

    @property
    def fixed_cross_obstacle_trajectory_npz(self) -> Path:
        """固定跨障目标成功轨迹的NPZ数据文件。"""

        return self.candidate_dir / "fixed_cross_obstacle_trajectory.npz"

    @property
    def fixed_cross_obstacle_trajectory_csv(self) -> Path:
        """固定跨障目标逐状态关节角CSV表格。"""

        return self.candidate_dir / "fixed_cross_obstacle_trajectory.csv"

    @property
    def fixed_cross_obstacle_ik_diagnostics_csv(self) -> Path:
        """固定跨障目标逐yaw、逐IK种子的诊断CSV。"""

        return self.candidate_dir / "fixed_cross_obstacle_ik_diagnostics.csv"

    @property
    def fixed_cross_obstacle_rrt_diagnostics_csv(self) -> Path:
        """固定跨障目标多次RRT运行统计CSV。"""

        return self.candidate_dir / "fixed_cross_obstacle_rrt_diagnostics.csv"

    @property
    def fixed_cross_obstacle_best_ik_npz(self) -> Path:
        """固定目标搜索中位置误差最小的best-effort IK构型。"""

        return self.candidate_dir / "fixed_cross_obstacle_best_ik.npz"

    def validate_required_files(self) -> None:
        """在真正加载PyBullet前先给出清楚的缺失文件错误。"""

        required = [
            (self.robot_urdf, "机器人URDF主文件"),
            (self.robot_mesh_dir, "机器人mesh目录"),
            (self.suction_config, "吸盘功能坐标系配置"),
            (self.tower_collision_mesh, "铁塔碰撞网格"),
            (self.tower_visual_mesh, "铁塔显示网格"),
            (self.candidate_npz("foot1"), "base端候选点数据"),
            (self.candidate_npz("foot2"), "L8端候选点数据"),
        ]

        missing = [
            f"{path}（{description}）"
            for path, description in required
            if not path.exists()
        ]
        if missing:
            joined = "\n".join(missing)
            raise FileNotFoundError(f"缺少必要输入文件：\n{joined}")
