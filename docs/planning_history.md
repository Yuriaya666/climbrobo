# CLIMBROBO Planning Development History

## 1. Current goal

研究八轴双端磁吸爬塔机器人的单步运动：固定一个物理磁吸端，另一个端寻找满足附着、运动学、碰撞和轨迹约束的目标。当前只验证单步和有限诊断实验，不自动递推整塔爬行。

## 2. Robot and environment conventions

机器人真实磁吸端是`base_end`和`l8_end`，来自URDF和`suction_frames.yaml`。长度单位为米，候选点和附着线坐标使用铁塔STL全局坐标系。铁塔碰撞模型为`models/Tower.STL`（碰撞网格），显示模型为`models/Tower_visual.STL`（GUI显示网格）。

最早曾假设`foot1/base_end`只能对应第一个附着面、`foot2/l8_end`只能对应第二个附着面，且两个面大致互相垂直。该限制后来取消：`base_end`和`l8_end`均可尝试`surface1`或`surface2`。

## 3. Attachment geometry

`models/Adhension1.STL`（surface1几何输入）和`models/Adhension2.STL`（surface2几何输入）沿用已验证的表面选择、局部UV坐标、Polygon/MultiPolygon和表面外法向处理。附着区域按现有`shrink_distance_m=0.062 m`收缩，再生成不跨越空洞的连续中心线。旧文件`foot1_attach_lines.npz`（surface1连续附着线数据）和`foot2_attach_lines.npz`（surface2连续附着线数据）保留不重建。

## 4. Planner upgrades

当前单步流程为：连续attach line → vertical progress目标搜索 → yaw搜索 → multi-start IK → 终点碰撞检查 → Straight → RRT-Connect。轨迹以关节角和必要的base pose序列保存，可脱离规划重新回放。

## 5. Step 1 and Step 2

Step 1已验证：`base_end`支撑，`l8_end`运动，使用RRT_CONNECT，目标z约`0.252001 m`，上升约`0.1767 m`。

Step 2已验证：`l8_end`支撑，`base_end`运动，目标z约`0.252001 m`。变基座前后link位置连续性为数值误差级别，已记录过最大位置变化约`0 m`、最大姿态变化约`2.107e-8 rad`。

## 6. Cross-obstacle and position-only diagnosis

第三步固定跨障目标约为`[0.06999983, 0.0, 1.12199991] m`。正式位置+法向IK未找到合法终点，best position error约`0.144481 m`。独立position-only工作空间测试取消法向、碰撞、轨迹和RRT后，PyBullet best约`0.174845 m`，Differential Evolution十个seed的最好约`0.119347 m`，局部精修约`0.118956 m`，Sobol sanity约`0.203331 m`。这些结果强烈表明该目标位置已超出当前八个关节真实限位下的位置工作空间，失败不主要由RRT造成。

## 7. Physical end / surface decoupling

本轮新增surface语义兼容层：旧`foot1/foot2`文件标签映射为`surface1/surface2`，但规划接口显式区分support/moving physical end与support/target surface。`AttachLineSet`、候选点、轨迹保存格式兼容旧NPZ，并可保存surface元数据。两个同面实验均从同一个`successful_step2_trajectory.npz`（第二步最终状态轨迹数据）独立恢复。

同面吸盘中心距约束采用现有收缩参数推导：`2 * 0.062 = 0.124 m`，不是新增经验尺寸。

## 8. Same-surface experiments

Test A：`base_end`固定在surface1，`l8_end`固定目标到surface1。最近向上目标为`[0.00162858, 0.06999996, 1.12200019] m`，z约`1.122000 m`，上升约`0.870260 m`，中心距约`0.869999 m`。rebase位置不连续误差为`0 m`。正式IK最佳位置误差约`0.102982 m`，因此没有合法q_goal、没有Straight或RRT。

Test B：`l8_end`固定在surface2，`base_end`固定目标到surface2。最近向上目标为`[0.06999983, -0.0000000002, 1.12199991] m`，z约`1.122000 m`，上升约`0.870047 m`，中心距约`0.869999 m`。rebase位置不连续误差约`1.87e-16 m`。修复运动端为base时的串联链上界计算后，正式IK最佳位置误差约`0.101108 m`，因此没有合法q_goal、没有Straight或RRT。

## 9. Important generated files

历史结果包括：`successful_one_step_trajectory.npz`（第一步成功关节轨迹）、`successful_step2_trajectory.npz`（第二步成功关节轨迹）、`successful_two_step_trajectory.npz`（两步拼接轨迹）、`position_workspace_test.npz`（位置工作空间汇总）、`fixed_cross_obstacle_ik_diagnostics.csv`（固定跨障IK诊断）、`same_surface_test_A_diagnostics.csv`（同面Test A诊断）和`same_surface_test_B_diagnostics.csv`（同面Test B诊断）。成功轨迹文件只在实际获得合法q_goal和无碰撞轨迹时生成；本轮两个同面测试均未生成成功轨迹文件。

## 10. Current limitations and next work

当前同面实验只固定了跨障后第一个向上中心线目标，没有执行双surface自动最优搜索，也没有规划Step 4。正式IK仍是数值多起点方法，工作空间边界附近可能需要独立优化器交叉验证。后续应先检查两个表面上较低、可达的同面目标，再决定是否实现双surface候选比较和多步循环。

## 11. Task-driven morphology design

位置工作空间诊断确认当前样机失败不是单纯RRT预算问题，因此研究从“继续调当前8R规划器”转为“基于真实Tower任务反向设计机构”。本阶段只比较对称`6R = 3R + central body + 3R`和对称`8R = 4R + central body + 4R`，不引入7R或新的机械辅助机构。

当前8R真实单侧有效杆长约为`[0.164438, 0.096702, 0.187842, 0.160085] m`，中央本体安装间距约`0.446 m`。中央本体是`robots_model/meshes/L4.STL`（中央刚性本体真实mesh），不是Box proxy。后续碰撞审计取消统一`0.065 m`连杆半径，改用真实L3/L5、L2/L6、L1/L7 STL PCA横截面得到的三类per-link半径`[0.064328, 0.081199, 0.081942] m`，nominal安全膨胀为`0.005 m`。关节/电机代理单独使用原始半径`0.040 m`并带同样膨胀；局部STL测量显示更大的上界，但由于STL没有独立电机壳体且会混入出杆几何，`0.040 m`作为受Baseline关节间距约`0.096702 m`约束的audit envelope，不是最终电机CAD尺寸。两个末端继续使用`base_link.STL`和`L8.STL`（真实末端mesh），Tower继续使用`models/Tower.STL`（真实碰撞网格）。

## 12. First morphology search results

第一阶段使用真实surface1/surface2 attach lines生成四类局部任务，并按位置、法向、终点碰撞、轨迹逐层筛选。粗搜索设计边界为每根可设计杆长的`0.60~1.80`倍标称值，并限制绝对上限`0.38 m`。

`BASELINE_8R`（当前真实杆长）在四个首层上行目标中的最坏position-only误差为`0.118858 m`，与历史独立workspace结果的全局精修误差`0.118956 m`同量级。

关键surface2首层目标的独立DE+局部精修三个seed对BEST_6R、BEST_OPTIMIZED_8R和有限轴8R都重复得到亚微米级位置误差，而BASELINE稳定在`0.118858 m`；因此杆长/轴架构带来的位置层改善不是单一起点局部解。

位置层粗优化得到：

* `BEST_6R`：删除最末端的一对镜像关节（`6r_drop_pair_3`），杆长`[0.276727, 0.173841, 0.310042] m`，四个代表目标position误差约`3.0e-11 m`以内；但正式position+normal多起点验证没有形成稳定合法终点。
* `BEST_OPTIMIZED_8R`：杆长`[0.294479, 0.158656, 0.209837, 0.208454] m`，四个代表目标position误差约`4.2e-11 m`以内；当前固定URDF轴架构下仍未稳定满足position+normal。

## 13. Discrete axis architecture follow-up

由于仅改杆长的第一阶段不能稳定通过法向层，启动有限离散轴架构诊断。一个镜像对称的`yaw-pitch-yaw-pitch` 8R候选在独立seed下四个首层目标均找到position+normal构型，说明当前失败不完全是杆长问题，轴架构会改变法向工作空间。

但把该候选放入旧碰撞代理后，四条Step 2之后的首边均在终点碰撞检查失败，最大已记录负间隙约`-0.0653 m`，典型断点为`central_body ↔ right_link_3`或末端mesh与杆件自碰撞。该结论随后被独立Collision Audit拆分：原始代理包含系统性误报，不能直接作为8R机构不可行的最终证据。

粗接触图使用`0.5 m`间隔而非毫米级离散，surface1约`136`个节点、surface2约`133`个节点。首个跨障区域约为`z=1.122 m`，当前候选在完整碰撞条件下的已验证最大高度仍为Step 2末端约`0.251953 m`。

## 14. Design diagnostics and engineering status

推荐候选在单个关键surface2首层任务上做了杆长`±10%`和目标位置`±5 mm`扰动，共`35`个position+normal样本全部通过；这是运动学/姿态鲁棒性结果，不包含碰撞鲁棒性。

基于当前URDF质量映射到参数化杆长的准静态重力力矩估计最大约`12.18 N·m`。项目没有电机连续或峰值输出力矩上限，因此状态为`ESTIMATE_ONLY_NO_MOTOR_LIMITS`，不能作为最终工程设计通过结论。

## 15. Current design status and next work

当前最有信息量的候选是`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`（优化杆长加有限轴架构），但它仍被中央本体/杆件碰撞淘汰。`BEST_6R`和固定当前轴架构的`BEST_OPTIMIZED_8R`尚未通过完整首边。下一步应在有限可制造轴架构内加入碰撞间隙作为分层筛选条件，再对通过终点的候选运行Straight和RRT-Connect，最后才进行连续contact-state graph和整塔路线验证。没有合法完整路线前，不应推荐6R或8R为最终机械设计。

## 16. Independent collision audit

暂停机构搜索后，对固定的`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`（有限轴8R候选）进行了独立Collision Audit。真实URDF在初始、Step 1终态和Step 2终态的非相邻自碰撞均为`0/3`；参数化FK与真实关节frame位置最大误差约`1.02e-7 m`，因此中央`L4` body frame变换没有发现错误。

审计确认`right_link_3`表示`J7→J8`，不是中央本体直接连接段；其 capsule 起止点严格等于J7/J8关节中心，当前有效长度`0.209837 m`。旧代理把原始`0.065 m`同时用于连杆和关节球，再加`0.005 m`膨胀，导致Baseline两个`joint_2↔joint_3`出现约`-0.043298 m`伪穿透，分类为`PROXY_TOO_CONSERVATIVE`。同时删除了右侧`J8→吸盘偏置`的重复capsule，只保留真实`L8.STL`末端网格，并修正碰撞查询和末端接触豁免；修正后真实URDF和参数化Baseline三个状态均为`0`个非相邻自碰撞。

四个正式position+normal q_goal已用固定候选、`seed=99`、10个multi-start、`normal_max_nfev=400`重新生成并立即保存到`models/design_results/q_goals/`（四个单任务NPZ和一个汇总NPZ）。旧代理修正前碰撞通过数为`0/4`，修正后nominal仍为`0/4`；`nominal+5 mm`和`nominal+10 mm`也均为`0/4`。修正后Task 0主要剩余`left_link_1↔left_link_3`自碰撞，其他任务主要为Tower碰撞，四个任务没有合法q_goal，因此没有运行Straight或RRT。

旧报告的`central_body↔right_link_3`另用历史`seed=20260815` Task 0构型独立复现：原始代理显示`-65.265 mm`，但J7→J8中心线在近零半径下对真实`L4.STL`仍有`+3.735 mm`间隙，故分类为`PROXY_TOO_CONSERVATIVE`，不是TRANSFORM_ERROR或中心线真实穿透。该历史参考保存在`legacy_central_right_audit.json`（旧central-right碰撞分类报告）和`legacy_seed_20260815_task0_q.npz`（历史诊断q构型）中。审计主产物为`collision_audit.json`（基线、四个q_goal及三档安全余量报告）、`collision_audit_pairs.csv`（近碰撞pair明细），GUI入口为`playback_collision_audit.py`（支持Baseline或q_goal可视化）。

## 17. Collision proxy calibration and collision-aware IK

固定`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`后，新增`proxy_calibration.py`（真实STL尺寸标定）、`collision_aware_ik.py`（多分支终点搜索）和`plot_collision_model.py`（碰撞体PNG/PyBullet可视化）。Baseline真实URDF与校准proxy在initial、Step 1终态和Step 2终态均为`0`个非相邻self-collision；两侧terminal capsule重复计数已移除，FK和末端位姿不变。

四个历史单一q_goal在校准proxy下仍为`0/4`，但这不再被解释为目标不可行。新的固定任务搜索每个任务执行`96`次IK尝试，得到`17/9/12/23`个unique position+normal branch，碰撞过滤后分别保留`6/3/3/6`个合法endpoint。两个任务Straight成功，两个任务使用未扩大预算的RRT-Connect成功。合法终点保存在`models/design_results/collision_free_q_goals/`（多分支碰撞筛选得到的终点文件），对应轨迹保存在`models/design_results/trajectories/`（实际Straight/RRT轨迹）。完整结果在`docs/collision_proxy_and_collision_aware_ik.md`（独立碰撞代理与多分支IK报告），尺寸表在`models/design_results/collision_proxy_dimensions.csv`（STL标定尺寸表），图片在`models/design_results/plots/`（当前8R碰撞模型视图）。

本轮只验证固定8R候选的首层局部任务，没有重新优化杆长、搜索新axis、研究6R或执行整塔规划。关节/电机独立CAD和最终工程碰撞包络仍未闭合，因此当前结论是“校准参数化proxy下存在合法首层终点和轨迹”，不是最终机械制造认证。

## 18. Autonomous whole-tower morphology checkpoint

在固定`OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`（当前有限轴8R候选）和当前真实Mesh/校准proxy下，继续完成了双候选接触图与精细轨迹验证。`BEST_6R`（位置层最佳6R候选）在Step 2初始接触状态即以`POSITION_WORKSPACE`失败，position error为`0.0313820116 m`，最大高度为`0 m`。

8R双候选接触图完成`133`次扩展，得到`210`个状态和`413`条边，最高endpoint高度`37.8549987748 m`，目标高度为`37.8594834094 m`，已进入`0.02 m`目标容差。该结果证明endpoint graph可到达塔顶区域，但不等于完整轨迹成功。

最高endpoint路线的rebase-aware fine validation前`10`条边均以Straight成功，随后在实际支持姿态下的目标`surface1:46:0.420261`（精细路线第11条边目标）没有得到标准或expanded IK候选。该失败发生在有效q_goal进入轨迹阶段之前，不是RRT预算问题；当前固定RRT参数没有在该路线被调用。最终 bounded-search 状态为`NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`，不宣称所有可能8R接触采样和所有未来机构设计均不可行。

完整研究报告为`docs/morphology_research_report.md`（本轮自主机构研究最终报告），最终接触图checkpoint为`models/design_results/checkpoints/whole_tower_search_checkpoint.json`（8R双候选图和精细路线断点）。后续若恢复研究，优先保留rebase后的多个IK分支并围绕第11条边扩展目标候选，不应先扩大RRT预算。
