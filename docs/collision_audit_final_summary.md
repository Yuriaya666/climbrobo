**1. 原 collision 0/4 是否可信？**

不完全可信。Baseline 真实 URDF 为 `0/3` 非相邻自碰撞，但旧代理为 `2/2/2`，存在系统性误报。因此旧结论已被标记为 `OLD_COLLISION_RESULT_INVALIDATED_BY_PROXY_BUG`。

**2. 代理修正**

- 旧 joint sphere 有效半径：`70 mm`，导致相邻关节约 `43.298 mm` 穿透。
- 改为独立 joint proxy：原始 `40 mm` + `5 mm` inflation。
- 删除重复的右端 `J8→吸盘偏置 capsule`，保留真实 `L8.STL`。
- 修正 Tower 接触豁免，`right_link_3` 不再被当作吸盘接触区域。

Baseline 修正后：真实 URDF `0/3`，参数化代理 `0/3`。

**3. `right_link_3`**

`right_link_3 = J7→J8`，不是中央本体直接连接杆。

- 运动学长度：`0.209837 m`
- capsule 起点/终点：严格等于 J7/J8 中心
- 原始 link radius：`0.065 m`
- safety inflation：`0.005 m`
- 有效半径：`0.070 m`

`0.065 m` 来自 `MorphologySpec.from_geometry()` 的人工默认值，不是电机 CAD，也不是吸盘尺寸。

**4. `central_body ↔ right_link_3`**

官方重新生成的四个 q_goal 中没有该碰撞对。

历史 seed 构型复现结果：

- capsule penetration：`-0.065265 m`
- J7→J8 中心线 clearance：`+0.003735 m`
- 分类：`PROXY_TOO_CONSERVATIVE`
- 不是 `TRUE_BODY_PENETRATION`
- 不是 `TRANSFORM_ERROR`

**5. 四个 q_goal**

正式 position+normal IK：`4/4` 成功，均已保存。

位置误差约为：

```text
1.19e-9 m
3.88e-13 m
1.28e-9 m
5.87e-12 m
```

法向误差均为 `0 deg`。

**6. 碰撞结果**

Nominal 修正代理：

```text
Task 0: -0.124463 m，left_link_1 ↔ left_link_3
Task 1: -0.068849 m，right_link_2 ↔ Tower
Task 2: -0.010690 m，left_link_1 ↔ Tower
Task 3: -0.044428 m，left_joint_1 ↔ Tower
```

- 修正前：`0/4`
- 修正后 nominal：`0/4`
- nominal +5 mm：`0/4`
- nominal +10 mm：`0/4`

没有合法 `q_goal`，因此没有运行 Straight 或 RRT。

**7. 最终分类**

`CASE E` 的碰撞未闭合状态：原始 central-right 失败主要是代理问题，但四个固定 q_goal 修正后仍被其他自碰撞或 Tower 碰撞淘汰。不能据此继续宣称“8R机构真实不可行”，也不能宣称已经可行。

**8. 修改与产物**

- [collision_proxy.py](/home/song/projects/climbrobo/environment/design/collision_proxy.py)（碰撞代理、终端几何和接触过滤修正）
- [collision_audit.py](/home/song/projects/climbrobo/environment/design/collision_audit.py)（Baseline、q_goal及三档安全余量审计）
- [morphology.py](/home/song/projects/climbrobo/environment/design/morphology.py)（分离link/joint proxy半径）
- [save_fixed_axis8_q_goals.py](/home/song/projects/climbrobo/environment/design/save_fixed_axis8_q_goals.py)（固定候选q_goal生成与保存）
- [legacy_central_right_audit.py](/home/song/projects/climbrobo/environment/design/legacy_central_right_audit.py)（历史central-right复现诊断）
- [playback_collision_audit.py](/home/song/projects/climbrobo/environment/design/playback_collision_audit.py)（支持q_goal GUI查看）
- [planning_history.md](/home/song/projects/climbrobo/docs/planning_history.md)（项目历史更新）
- [morphology_design_results.md](/home/song/projects/climbrobo/docs/morphology_design_results.md)（机构设计结果文档更新）

q_goal文件位于 [models/design_results/q_goals/](/home/song/projects/climbrobo/models/design_results/q_goals/)（四个正式q_goal及汇总文件）。

历史 central-right 报告为 [legacy_central_right_audit.json](/home/song/projects/climbrobo/models/design_results/diagnostics/legacy_central_right_audit.json)（历史代理碰撞分类报告）。

GUI命令：

```bash
.venv/bin/python -m environment.design.playback_collision_audit \
  --q-goal models/design_results/q_goals/optimized_8r_axis8_task_0.npz \
  --pair central_body,right_link_3 \
  --seconds 30
```

本轮未生成轨迹文件。
