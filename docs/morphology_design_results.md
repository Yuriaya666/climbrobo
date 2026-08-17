# CLIMBROBO Morphology Design Results

## 1. Problem definition

本阶段不再继续堆叠当前样机的RRT参数。目标是基于真实Tower、真实附着线、真实吸盘和中央本体，判断对称6R或对称8R能否从Step 2状态开始形成连续无碰撞爬行。

评价顺序为：结构有效性 → position-only → position+normal → 终点碰撞 → Straight → RRT-Connect → contact-state graph。单点workspace成功不能替代整塔路线成功。

## 2. Baseline robot

当前真实模型为`BASELINE_8R`：每侧4R，URDF关节为J1至J8，中央刚性link为L4。单侧从中央本体到吸盘的有效标称长度约为：

```text
[0.164438, 0.096702, 0.187842, 0.160085] m
```

中央本体左右安装间距约`0.446 m`。历史position-only关键目标`[0.06999983, 0, 1.12199991] m`的独立全局精修误差为`0.1189555 m`。

## 3. Symmetry and design space

只研究：

```text
6R = 3R + central body + mirrored 3R
8R = 4R + central body + mirrored 4R
```

6R通过删除当前4R侧链中一对镜像对应的旋转关节形成真正的3R序列，保留被删除关节的固定几何，不是把8R关节简单锁到零位。7R不属于本项目设计空间。

## 4. Parameterized collision model

中央本体使用`robots_model/meshes/L4.STL`（真实中央本体mesh）。Tower使用`models/Tower.STL`（真实碰撞网格）。两个物理末端使用现有`base_link.STL`和`L8.STL`（真实末端mesh）。新裸连杆Capsule改为按真实STL PCA横截面标定的per-link半径：镜像三类有效杆段约为`0.064328 m`、`0.081199 m`和`0.081942 m`，nominal安全膨胀为`0.005 m`。关节/电机使用独立原始半径`0.040 m`球形audit proxy并增加同样膨胀；该值受真实Baseline关节间距和局部STL测量约束，但不是独立电机CAD尺寸。左右两个末端由真实mesh表示，terminal capsule不重复计入。

这些代理只用于尚无新CAD时的第一阶段评估。它们没有关闭中央本体、吸盘、Tower或self collision。

## 5. Task Suite

任务从真实surface1/surface2连续attach lines生成，包括：

* surface1同面向上；
* surface2同面向上；
* surface1到surface2；
* surface2到surface1；
* `base_end`支撑和`l8_end`支撑两种角色；
* Step 2之后首个障碍后附着区域，目标约`z=1.122 m`。

粗接触图以`0.5 m`间隔生成，surface1约136个节点、surface2约133个节点；连续`s`只在局部目标细化时使用。

## 6. First-stage length results

### BASELINE_8R

当前杆长下，四个首层任务的position-only最坏误差约`0.118858 m`，未达到目标位置，因此首边在workspace层断开。

### BEST_6R

粗优化最佳拓扑为：

```text
topology: 6r_drop_pair_3
deleted pair: 最末端一对镜像关节
L1: 0.276727 m
L2: 0.173841 m
L3: 0.310042 m
```

四个代表目标position-only误差约`3.0e-11 m`以内，说明该3R侧链在位置层有足够长度。但在固定当前轴架构下，position+normal多起点结果不稳定，尚未形成可碰撞验证的合法终点。

### BEST_OPTIMIZED_8R

保持当前4R topology，只优化杆长：

```text
L1: 0.294479 m
L2: 0.158656 m
L3: 0.209837 m
L4: 0.208454 m
```

四个代表目标position-only误差约`4.2e-11 m`以内；固定当前轴架构时，法向层仍无法稳定同时满足位置和法向。

关键surface2首层目标的独立DE+局部精修交叉验证中，三个seed分别得到：BASELINE约`0.118858 m`，BEST_6R约`9.0e-11~3.9e-10 m`，BEST_OPTIMIZED_8R约`5.2e-12~4.4e-10 m`；有限轴8R约`5.8e-12~3.7e-11 m`。这确认优化候选的位置层结果不是单一起点局部解。

## 7. Finite axis architecture results

有限离散轴诊断中，`yaw-pitch-yaw-pitch`镜像8R与上述优化杆长组合，在独立seed下四个首层目标均找到position+normal构型。该结果说明额外轴架构可以改善法向能力。

旧碰撞代理终点验证为`0/4`首边通过。典型历史失败：

```text
central_body ↔ right_link_3 self collision
minimum clearance ≈ -0.0653 m
```

独立Collision Audit发现旧代理还存在相邻关节球误报和右侧终端偏置重复capsule。修正代理并复用重新保存的四个正式q_goal后，nominal、nominal+5 mm、nominal+10 mm仍均为`0/4`终点通过；当前剩余失败已分离为Task 0的非相邻杆件代理自碰撞，以及其他Task的Tower碰撞。因四个终点仍不合法，没有调用Straight或RRT。

## 8. Whole-tower contact-state planning

当前已建立粗粒度contact node数据结构和首边验证入口。对最终候选，Step 2末端高度约`0.251953 m`；第一个障碍后目标约`1.122000 m`。当前候选在终点碰撞层就断开，因此尚未生成完整整塔路线，也没有把局部position成功写成whole-tower success。

## 9. Robustness and sensitivity

对有限轴8R候选，在surface1到surface2的关键目标上测试杆长因子`0.90、0.95、1.00、1.05、1.10`以及目标位置三个方向的`±5 mm`扰动，共35个position+normal样本均通过。该结果只覆盖运动学和法向，不覆盖Tower/self collision；碰撞裕量仍是未解决风险。

## 10. Torque analysis

按当前URDF质量对参数化杆长进行标称线密度映射，在一个代表性首层构型下估算最大重力力矩约`12.18 N·m`。项目没有电机连续/峰值输出力矩上限，因此状态为：

```text
ESTIMATE_ONLY_NO_MOTOR_LIMITS
```

这不是最终机械设计的力矩通过结论。后续必须补充电机、减速器、杆件和吸盘的质量及额定/峰值力矩。

## 11. Recommended morphology

当前不能推荐“已经完成整塔”的6R或8R。若只按运动学/法向层继续研究，优先保留`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`作为下一轮碰撞间隙优化候选；它比6R更有能力满足四类首层位置和法向任务，但目前仍被中央本体和杆件自碰撞淘汰。

## 12. Remaining engineering risks

* Capsule半径和安全膨胀需要用新机构CAD复核；当前per-link STL包络和`0.040 m`关节半径仍是参数化审计代理，不是新杆件/电机最终形状。
* 中央本体真实mesh会直接限制折叠构型，不能用缩小Box替代。
* 终点合法后还必须重新运行Straight和多seed RRT-Connect，并验证整段轨迹碰撞。
* 当前没有完整接触状态配置和电机力矩上限，不能输出最终制造尺寸。

## 13. Result files

主要结果位于`models/design_results/`（机构设计搜索结果目录）：

* `baseline_8r.json`（当前真实8R基线结果）；
* `best_6r.json`（位置层最佳6R候选）；
* `best_8r.json`（位置层最佳优化8R候选）；
* `best_axis8.json`（有限轴架构8R候选）；
* `design_search_summary.csv`（设计搜索汇总表）；
* `task_suite_results.csv`（真实附着线任务集摘要）；
* `whole_tower_routes/`（首个contact-state edge验证报告目录）；
* `diagnostics/`（鲁棒性、力矩和Collision Audit结果目录）；
* `q_goals/`（固定有限轴8R四个正式position+normal q_goal文件）。
* `collision_free_q_goals/`（固定8R多分支搜索得到的合法终点文件）。
* `collision_proxy_dimensions.csv`（按真实STL标定的碰撞代理尺寸表）。
* `plots/collision_model_optimized_8r_isometric.png`和`plots/collision_model_optimized_8r_side.png`（碰撞体可视化）。

GUI查看候选构型的命令：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m environment.design.playback_design --design axis8 --task-index 1 --keep-open
```

查看固定q_goal及J7→J8碰撞几何：

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m environment.design.playback_collision_audit \
  --q-goal models/design_results/q_goals/optimized_8r_axis8_task_0.npz \
  --pair central_body,right_link_3 --seconds 30
```

## 14. Collision audit status

在继续使用有限轴8R碰撞结果前，已对真实基线和参数化代理做独立审计。真实URDF的初始、Step 1终态和Step 2终态均无非相邻自碰撞；参数化FK与真实关节frame最大位置误差约`1.02e-7 m`。`right_link_3`明确是`J7→J8`段，不是中央本体直接安装段。

审计同时发现旧关节Sphere有效半径`0.070 m`大于相邻关节中心距约`0.096702 m`的一半，导致参数化基线的`joint_2↔joint_3`产生约`-0.043298 m`伪穿透，分类为`PROXY_TOO_CONSERVATIVE`。另确认历史`central_body↔right_link_3`失败来自capsule包络：J7→J8中心线仍有`+3.735 mm`间隙。修正了关节/连杆代理索引、右端重复终端capsule和接触豁免范围；没有关闭中央本体碰撞，也没有永久放行`central_body↔right_link_3`。四个正式q_goal已保存并完成修正前后及三档安全余量复验。详见`models/design_results/diagnostics/collision_audit.json`（碰撞审计报告）、`models/design_results/diagnostics/legacy_central_right_audit.json`（历史central-right分类报告）和`models/design_results/q_goals/`（四个正式q_goal文件）。
审计同时发现旧关节Sphere有效半径`0.070 m`大于相邻关节中心距约`0.096702 m`的一半，导致参数化基线的`joint_2↔joint_3`产生约`-0.043298 m`伪穿透，分类为`PROXY_TOO_CONSERVATIVE`。另确认历史`central_body↔right_link_3`失败来自capsule包络：J7→J8中心线仍有`+3.735 mm`间隙。修正了关节/连杆代理索引、两侧重复terminal capsule和接触豁免范围；没有关闭中央本体碰撞，也没有永久放行`central_body↔right_link_3`。四个正式q_goal已保存并完成修正前后及三档安全余量复验。详见`models/design_results/diagnostics/collision_audit.json`（碰撞审计报告）、`models/design_results/diagnostics/legacy_central_right_audit.json`（历史central-right分类报告）和`models/design_results/q_goals/`（四个正式q_goal文件）。

## 15. Collision-aware endpoint follow-up

固定`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`后，按真实STL标定per-link proxy，并对四个首层目标各执行`96`次IK尝试。四个任务分别得到`17/9/12/23`个unique position+normal branch，碰撞过滤后分别保留`6/3/3/6`个合法endpoint。两个任务Straight成功，另外两个任务在未扩大预算的RRT-Connect下成功；因此历史四个单一q_goal的`0/4`不能代表这些目标没有其他合法IK构型。完整数据见`docs/collision_proxy_and_collision_aware_ik.md`（本轮独立报告）、`models/design_results/diagnostics/collision_proxy_and_collision_aware_ik.json`（结构化诊断）和`models/design_results/collision_free_q_goals/`（合法终点文件）。
