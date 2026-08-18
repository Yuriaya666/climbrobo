# Morphology Research Status

## Status

`NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`（固定候选、双候选接触图和 fine trajectory 验证已完成）

## Research objective

在真实 Tower、真实附着线、真实中央本体和当前碰撞代理下，判断对称 6R 与对称 8R 是否存在从塔底到目标高度的连续交替支撑路线。最终状态只允许为：

* `SUCCESS_6R`
* `SUCCESS_8R_REQUIRED`
* `NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`

计算或会话中断不能作为研究结论，必须保留 `INTERRUPTED_RESUMABLE` 和 resume 命令。

## Fixed research constraints

* 只研究 symmetric 6R / symmetric 8R；不引入 7R、辅助机构、伸缩机构或新 Tower 几何。
* 当前 8R 固定为 `OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`，杆长为 `[0.2944788185, 0.1586560034, 0.2098367564, 0.2084544616] m`。
* 中央本体使用真实 `L4.STL`，Tower 使用真实 `Tower.STL`，吸盘使用现有真实末端几何。
* 8R link proxy 使用真实 STL 标定的 per-link 半径 `[0.064328, 0.081199, 0.081942, 0.062050] m`，关节原始 proxy 半径为 `0.040 m`，nominal inflation 为 `0.005 m`。
* RRT 固定为每条边最多 1200 次迭代、2 个 seed；不通过扩大 RRT 预算替代机构或接触图验证。

## Completed evidence

### 6R

`BEST_6R`（删除末端镜像关节对，杆长 `[0.276727, 0.173841, 0.310042] m`）已完成初始接触状态验证。初始目标 position error 为 `0.0313820116 m`，normal error 为 `1.053414 deg`，因此在 `POSITION_WORKSPACE` 层失败，最大爬升高度为 `0 m`。结果见 `models/design_results/whole_tower_routes/best6r_full_search.json`（6R 全塔验证报告）和对应 checkpoint。

### 8R 局部与碰撞审计

固定轴 `YAW-PITCH-YAW-PITCH` 的四类首层 position+normal 任务均可找到合法 IK。旧单分支 `0/4` 结论被 collision proxy 假碰撞审计修正；多分支搜索已找到 4 个任务的 collision-free endpoint，并有 Straight/RRT 局部轨迹证据。详见 `docs/collision_proxy_and_collision_aware_ik.md`（碰撞代理标定与多分支 IK 报告）。

### 8R 接触图

旧的单候选接触图曾达到约 `37.855 m` endpoint height，但没有证明 trajectory connectivity。随后发现上游分支和目标顺序会改变后续 rebase，已加入 branch rescue 和多候选接触图搜索。

当前双候选搜索 checkpoint：

* `models/design_results/checkpoints/whole_tower_search_checkpoint.json`（当前 8R 双候选接触图断点）
* 双候选搜索已完成 `133` 次扩展，`210` 个状态，`413` 条边，最高 endpoint height `37.8549988 m`；目标高度 `37.8594834 m`，进入 `0.02 m` endpoint 容差。
* fine validation 的最高路线前 `10` 条边均 Straight 成功；在实际 rebase 后的第 `11` 条边目标 `surface1:46:0.420261` 没有返回标准或 expanded IK branch，未形成完整连续轨迹。

## Completed command

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m environment.design.whole_tower_search \
  --design axis8 \
  --max-expansions 300 \
  --candidates-per-surface 2 \
  --resume
```

该命令已完成接触图和最高路线 fine trajectory validation。固定 RRT 配置为每条边最多 `1200` 次迭代、`2` 个 seed；本次最高路线在产生合法 re-based q_goal 前停止，因此没有新增 RRT 调用。不要据此扩大 RRT 预算或重新启动机构大搜索。

## Stop condition

6R 已在初始接触位置层失败；8R endpoint graph 到达塔顶容差，但 bounded fine trajectory search 未找到完整路线。因此本轮最终状态为 `NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`。该状态只适用于已执行的固定候选和搜索空间，不等价于所有未来 8R 形态或所有接触采样均不可行。完整报告见 `docs/morphology_research_report.md`（本轮自主机构研究报告）。
