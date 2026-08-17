# Collision Proxy Calibration and Collision-Aware IK

## 1. Background

本报告固定 `OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`，不重新优化杆长、轴架构或整塔路径。旧 `65 mm` 统一 link radius 来自人工默认值；同时，单个 position+normal q_goal 碰撞不能证明同一目标没有其他 IK branch。

## 2. Baseline Collision Audit

Baseline sanity status: `PASS`。真实 URDF 非相邻 self-collision counts: `[0, 0, 0]`；校准 proxy counts: `[0, 0, 0]`。

| state | real URDF bad pairs | calibrated proxy bad pairs |
|---|---:|---:|
| initial | 0 | 0 |
| step1_final | 0 | 0 |
| step2_final | 0 | 0 |

左端 terminal capsule 已从碰撞代理显示/碰撞体中移除，因为 `base_link.STL` 已经代表该物理末端结构；不改变 FK 或吸盘中心。

## 3. Real Geometry Measurement

连杆主轴由 STL PCA 最大特征方向确定；连杆 raw radius = PCA 横向两个 extent 的最大值的一半。关节没有独立电机 CAD，因此先测量 child-link 原点 `55 mm` 邻域内顶点距离的 `99th percentile`，再用真实基线关节间距审计得到的 `40 mm raw` 上限作为独立 joint sphere。该值是可解释的 audit envelope，不冒充电机 CAD；最终电机定型仍需独立外壳模型。

| mesh | axial length (m) | cross width (m) | cross height (m) | equivalent radius (m) |
|---|---:|---:|---:|---:|
| base_link.STL | 0.19734 | 0.120071 | 0.124098 | 0.0620492 |
| L1.STL | 0.241807 | 0.162908 | 0.0963869 | 0.0814542 |
| L2.STL | 0.17605 | 0.162398 | 0.148289 | 0.081199 |
| L3.STL | 0.209216 | 0.0985967 | 0.128656 | 0.0643282 |
| L4.STL | 0.419912 | 0.447327 | 0.201918 | 0.223664 |
| L5.STL | 0.209216 | 0.0985967 | 0.128656 | 0.0643282 |
| L6.STL | 0.17605 | 0.162392 | 0.14829 | 0.0811961 |
| L7.STL | 0.241553 | 0.163884 | 0.0964291 | 0.0819419 |
| L8.STL | 0.197344 | 0.120079 | 0.124099 | 0.0620495 |

## 4. Final Collision Proxy Definition

- central body：真实 `L4.STL` mesh；
- link body：Capsule，使用每一类镜像 STL 的 per-link radius，杆长随当前 MorphologySpec 变化；
- joint/motor：独立 Sphere，使用局部 STL 邻域 profile；
- endpoint/suction：`base_link.STL` 与 `L8.STL` 真实 mesh；
- safety inflation：nominal `+5 mm`，另测试 `+10 mm` 和 `+15 mm` 总膨胀；

| component | source | nominal radius (m) | effective radius (m) |
|---|---|---:|---:|
| link_1 | L3.STL + L5.STL | 0.0643282 | 0.0693282 |
| link_2 | L2.STL + L6.STL | 0.081199 | 0.086199 |
| link_3 | L1.STL + L7.STL | 0.0819419 | 0.0869419 |
| terminal | base_link.STL + L8.STL | 0.0620495 | 0.0670495 |
| joint_1 | L4.STL + L5.STL | 0.04 | 0.045 |
| joint_2 | L3.STL + L6.STL | 0.04 | 0.045 |
| joint_3 | L2.STL + L7.STL | 0.04 | 0.045 |
| joint_4 | L1.STL + L8.STL | 0.04 | 0.045 |

## 5. Collision Model Visualization

- `/home/song/projects/climbrobo/models/design_results/plots/collision_model_optimized_8r_isometric.png`
- `/home/song/projects/climbrobo/models/design_results/plots/collision_model_optimized_8r_side.png`

## 6. Existing Four q_goal Re-evaluation

已读取 `/home/song/projects/climbrobo/models/design_results/q_goals` 中保存的四个 q_goal；没有重新 IK。

| variant | collision pass | task 0 | task 1 | task 2 | task 3 |
|---|---:|---|---|---|---|
| nominal | 0/4 | left_link_1↔left_link_3 (-0.140826 m) | right_link_2↔Tower (-0.0840917 m) | left_link_1↔Tower (-0.010122 m) | left_joint_1↔Tower (-0.0444278 m) |
| nominal_plus_5mm | 0/4 | left_link_1↔left_link_3 (-0.148751 m) | right_link_2↔Tower (-0.0890917 m) | left_link_1↔Tower (-0.0142622 m) | left_joint_1↔Tower (-0.0494278 m) |
| nominal_plus_10mm | 0/4 | left_link_1↔left_link_3 (-0.152754 m) | right_link_2↔Tower (-0.0940917 m) | left_link_1↔Tower (-0.0181851 m) | left_joint_1↔Tower (-0.0544278 m) |

## 7. Collision-Aware IK Search Method

每个固定目标执行 yaw-conditioned full-pose seeds 与 normal-only seeds；所有候选最终只按正式 `position <= 5 mm`、`normal <= 3 deg`、joint limits 和 support constraint 判定。通过 position+normal 的候选先去重，再逐个进入真实 Tower + self collision checker；合法终点按 minimum clearance 和 joint-limit margin 排序。若普通 multi-start 无合法终点，最多对最接近的若干 branch 做有限的 collision-aware L-BFGS-B 局部 refinement。报告中的 `0.05 m` endpoint clearance 表示在 `50 mm` 近距离查询窗口内没有更近的非接触 pair，即 `>= 50 mm` 的下界，不是无限远。

## 8. Task 0 Results

- task: `surface1_to_surface1_base_end_to_l8_end`
- target xyz: `[0.0016285776666451288, 0.06999996003949106, 1.122000185488565]`
- target normal: `[-0.9999999958856641, -7.264577108190293e-08, -9.071199780455458e-05]`
- IK attempts: `96`
- position-normal converged: `17`
- unique IK branches: `17`
- collision-free endpoints: `6`
- dominant rejected pairs: `{'left_link_1↔Tower': 3, 'left_joint_1↔Tower': 1, 'right_link_1↔Tower': 4, 'right_link_2↔Tower': 2, 'right_joint_2↔Tower': 2, 'central_body↔left_link_3': 4, 'central_body↔left_joint_4': 4, 'left_link_1↔left_link_3': 4, 'left_link_1↔left_joint_3': 4, 'left_link_3↔left_joint_1': 4, 'left_joint_1↔left_joint_4': 4, 'central_body↔right_link_3': 2, 'central_body↔right_joint_4': 1, 'l8_end_mesh↔right_joint_1': 1, 'right_link_1↔right_link_3': 3, 'right_link_1↔right_joint_3': 3, 'right_link_3↔right_joint_1': 2, 'right_joint_1↔right_joint_4': 2}`
- refinement attempts: `0`
- best collision-free clearance: `0.05 m`
- best joint-limit margin: `0.512675 rad`
- best q: `[1.299912975625492, -0.8245861019750699, 0.48732541534393603, 0.8327901363947405, 1.0292413767830542, -0.8639406328946639, 0.3988348386025586, 0.5591122799200753]`
- collision-free q_goal artifact: `{'path': '/home/song/projects/climbrobo/models/design_results/collision_free_q_goals/optimized_8r_axis8_task_0.npz', 'task_index': 0}`
- Straight: `True`
- RRT-Connect: `False`
- trajectory artifact: `{'npz': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface1_to_surface1_base_end_to_l8_end.npz', 'csv': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface1_to_surface1_base_end_to_l8_end.csv', 'method': 'Straight', 'minimum_clearance_m': 0.02, 'planning_time_s': 2.3248893739655614}`

## 9. Task 1 Results

- task: `surface1_to_surface2_base_end_to_l8_end`
- target xyz: `[0.06999982686116368, -2.0730049044290056e-10, 1.1219999082207226]`
- target normal: `[2.7755572978987915e-14, -1.0, 1.4210853365241813e-11]`
- IK attempts: `96`
- position-normal converged: `9`
- unique IK branches: `9`
- collision-free endpoints: `3`
- dominant rejected pairs: `{'left_link_1↔Tower': 5, 'right_link_1↔Tower': 5, 'right_link_2↔Tower': 4, 'left_joint_1↔Tower': 4, 'right_joint_2↔Tower': 4, 'left_link_1↔left_link_3': 2, 'left_link_1↔left_joint_3': 2}`
- refinement attempts: `0`
- best collision-free clearance: `0.05 m`
- best joint-limit margin: `0.792773 rad`
- best q: `[0.8133482376733606, -1.45439925970133, -0.20722736586076373, 1.4334514218720138, -1.4956541117303042, 0.9062154608864894, 0.13923344344605307, -1.7839000754586616]`
- collision-free q_goal artifact: `{'path': '/home/song/projects/climbrobo/models/design_results/collision_free_q_goals/optimized_8r_axis8_task_1.npz', 'task_index': 1}`
- Straight: `True`
- RRT-Connect: `False`
- trajectory artifact: `{'npz': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface1_to_surface2_base_end_to_l8_end.npz', 'csv': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface1_to_surface2_base_end_to_l8_end.csv', 'method': 'Straight', 'minimum_clearance_m': 0.02, 'planning_time_s': 2.329832734540105}`

## 10. Task 2 Results

- task: `surface2_to_surface1_l8_end_to_base_end`
- target xyz: `[0.0016285776666451288, 0.06999996003949106, 1.122000185488565]`
- target normal: `[-0.9999999958856641, -7.264577108190293e-08, -9.071199780455458e-05]`
- IK attempts: `96`
- position-normal converged: `12`
- unique IK branches: `12`
- collision-free endpoints: `3`
- dominant rejected pairs: `{'left_link_1↔Tower': 6, 'right_link_1↔Tower': 4, 'right_joint_1↔Tower': 4, 'right_link_1↔right_link_3': 2, 'left_link_2↔Tower': 4, 'left_joint_2↔Tower': 2}`
- refinement attempts: `0`
- best collision-free clearance: `0.05 m`
- best joint-limit margin: `0.88437 rad`
- best q: `[-1.251374595965432, 1.3412232726069693, -0.016655253539622127, -1.826586849841097, 0.7283729534255187, -1.0480787376076022, -0.11562966535761621, 1.2660395666697033]`
- collision-free q_goal artifact: `{'path': '/home/song/projects/climbrobo/models/design_results/collision_free_q_goals/optimized_8r_axis8_task_2.npz', 'task_index': 2}`
- Straight: `False`
- RRT-Connect: `True`
- trajectory artifact: `{'npz': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface2_to_surface1_l8_end_to_base_end.npz', 'csv': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface2_to_surface1_l8_end_to_base_end.csv', 'method': 'RRT-Connect', 'minimum_clearance_m': 0.004317280004870347, 'planning_time_s': 3.7460304144769907}`

## 11. Task 3 Results

- task: `surface2_to_surface2_l8_end_to_base_end`
- target xyz: `[0.06999982686116368, -2.0730049044290056e-10, 1.1219999082207226]`
- target normal: `[2.7755572978987915e-14, -1.0, 1.4210853365241813e-11]`
- IK attempts: `96`
- position-normal converged: `23`
- unique IK branches: `23`
- collision-free endpoints: `6`
- dominant rejected pairs: `{'left_link_1↔Tower': 6, 'right_link_1↔Tower': 7, 'left_joint_1↔Tower': 2, 'left_link_1↔left_link_3': 6, 'right_joint_1↔Tower': 1, 'central_body↔left_link_3': 4, 'left_link_1↔left_joint_3': 5, 'left_link_3↔left_joint_1': 4, 'right_link_2↔Tower': 2, 'right_joint_2↔Tower': 2, 'right_link_1↔right_joint_3': 5, 'left_link_2↔Tower': 3, 'central_body↔right_link_3': 3, 'central_body↔right_joint_4': 2, 'right_link_1↔right_link_3': 4, 'right_link_3↔right_joint_1': 3, 'right_joint_1↔right_joint_4': 3, 'left_joint_2↔Tower': 1, 'central_body↔left_joint_4': 3, 'left_joint_1↔left_joint_4': 2}`
- refinement attempts: `0`
- best collision-free clearance: `0.05 m`
- best joint-limit margin: `0.87067 rad`
- best q: `[1.4694515756222672, -0.4576497248957253, 0.129329569539694, 0.6057219104408001, 1.5047883501908268, -1.0494871797739755, 0.07635978974583872, 0.9552409495170887]`
- collision-free q_goal artifact: `{'path': '/home/song/projects/climbrobo/models/design_results/collision_free_q_goals/optimized_8r_axis8_task_3.npz', 'task_index': 3}`
- Straight: `False`
- RRT-Connect: `True`
- trajectory artifact: `{'npz': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface2_to_surface2_l8_end_to_base_end.npz', 'csv': '/home/song/projects/climbrobo/models/design_results/trajectories/optimized_8r_axis8_surface2_to_surface2_l8_end_to_base_end.csv', 'method': 'RRT-Connect', 'minimum_clearance_m': 0.011171343816624393, 'planning_time_s': 3.4094625851139426}`

## 12. What the 8R Redundancy Actually Provides

本轮把同一个目标的多组 position+normal IK branch 分开统计。若 branch 数量大于1但 collision-free endpoint 为0，则冗余在运动学层存在，但在当前固定 Tower/central-body/proxy 几何下没有转化为合法终点；若有合法 endpoint，则其 clearance 和碰撞 pair 记录说明冗余实际绕开了什么。

## 13. Current Conclusion

CASE A：至少一个 position+normal 合法终点完成了无碰撞轨迹。

## 14. Generated Files

- `/home/song/projects/climbrobo/docs/collision_proxy_and_collision_aware_ik.md`（本轮完整独立报告）
- `/home/song/projects/climbrobo/models/design_results/q_goals`（四个历史 q_goal 输入目录）
- `/home/song/projects/climbrobo/models/design_results/collision_free_q_goals`（本轮多分支筛选得到的合法 endpoint）
- `/home/song/projects/climbrobo/models/design_results/trajectories`（仅在轨迹成功时生成）
- `models/design_results/collision_proxy_dimensions.csv`（碰撞代理尺寸表）
- `environment/design/proxy_calibration.py`（真实 STL 尺寸标定模块）
- `environment/design/collision_aware_ik.py`（固定候选多分支 IK 与审计脚本）
- `environment/design/plot_collision_model.py`（碰撞体 PNG/PyBullet 可视化脚本）

## 15. Recommended Next Step

先检查碰撞体 PNG 和本报告中的 branch/pair 统计；在确认 proxy 尺寸和 endpoint 搜索结论后，再决定是否进入局部机构设计修改。
