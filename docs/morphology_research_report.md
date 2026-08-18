# Autonomous Morphology Research Report

**Run date:** 2026-08-18  
**Final bounded-search status:** `NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`

## 1. Scope and fixed design space

This report closes the current autonomous morphology checkpoint. The evaluated design space was limited to the already selected candidates; no new link-length optimization, axis search, 6R redesign, Tower modification, suction modification, or RRT budget increase was performed during this continuation.

The fixed 8R candidate was `OPTIMIZED_8R_YAW_PITCH_YAW_PITCH`（本轮固定的 YAW-PITCH-YAW-PITCH 8R 设计）:

```text
axes:         YAW-PITCH-YAW-PITCH
link lengths: [0.2944788185, 0.1586560034, 0.2098367564, 0.2084544616] m
link radii:   [0.064328, 0.081199, 0.081942, 0.062050] m
joint radius: 0.040 m raw + 0.005 m nominal inflation
central body: real L4.STL mesh
Tower:        real Tower.STL mesh
```

The 6R comparison candidate was `BEST_6R`（位置层最佳对称 6R 候选）:

```text
topology: 6r_drop_pair_3
link lengths: [0.276727, 0.173841, 0.310042] m
```

## 2. Baseline and collision-model context

`BASELINE_8R`（当前真实样机基线） uses approximate effective per-side lengths `[0.164438, 0.096702, 0.187842, 0.160085] m`. Its historical position-only failure on the critical first obstacle region was approximately `0.118858 m`, consistent with the independently established workspace mismatch.

The old `0/4` endpoint collision result is not used as independent proof of 8R infeasibility. The collision audit found an oversized joint proxy and duplicate/incorrect proxy treatment. The current evaluator uses real central-body and Tower meshes, real suction-end geometry, per-link STL-calibrated capsules, independent joint proxies, and explicit parent/child collision handling.

## 3. 6R result

The best 6R candidate failed before contact-state graph expansion:

```text
initial position error: 0.0313820116 m
initial normal error:   1.053414 deg
failure:                POSITION_WORKSPACE
maximum height:         0 m
```

There was no valid initial contact state for the current Step 2 starting configuration. Therefore this candidate cannot produce a whole-Tower route under the current fixed design and task constraints.

Result file: `models/design_results/whole_tower_routes/best6r_full_search.json`（6R 初始接触状态与全塔验证报告）.

## 4. 8R endpoint contact graph

The resumed two-candidate-per-surface contact graph completed with:

```text
expansions:                 133
states:                     210
edges:                      413
goal height:                37.8594834094 m
maximum endpoint height:    37.8549987748 m
height difference to goal:  0.0044846346 m
endpoint goal tolerance:    0.020 m
```

The endpoint graph therefore reached the Tower-top tolerance region. This is a reachability result for contact endpoints only; it is not a continuous collision-free trajectory result.

Endpoint edge classification in the completed graph:

| Classification | Count |
|---|---:|
| Endpoint valid | 248 |
| `TOWER_COLLISION` | 60 |
| `NORMAL_WORKSPACE` | 52 |
| `POSITION_WORKSPACE` | 47 |
| `SELF_COLLISION` | 6 |

The coarse evaluator commonly reports `inf` for valid endpoint clearance because it does not retain a finite closest-distance value for every successful endpoint. Those values are not interpreted as a positive clearance margin.

The highest graph terminal was `state_0076_surface1_0_0.170000`（双候选接触图最高终点状态） at depth 76 and height `37.8549987748 m`.

## 5. Fine trajectory validation

The highest endpoint route was then validated with rebase-aware fine checking and the existing branch-rescue logic. The route diverged from the graph q sequence whenever a rescued IK branch changed the support pose; downstream states were rebuilt from that actual state.

The fine validator produced:

```text
successful Straight edges: 10
successful RRT edges:     0
RRT calls on this route:   0
rescue events:             8
verified prefix height:    5.7594844794 m
```

All ten successfully validated edges used joint-space Straight interpolation. The finite trajectory clearances reported for successful prefix edges were:

```text
0.01158936 m
0.01455853 m
0.00639607 m
0.00310668 m
0.01150153 m
0.01224384 m
```

The smallest finite reported clearance in the successful prefix was `0.00310668 m`（约 3.107 mm）. Some early edges returned `inf` from the existing checker; those entries are retained as reported and are not treated as measurable margin.

## 6. First fine-route failure

The first unresolved fine edge was:

```text
source state: state_0010_surface2_48_0.330946
source height: 5.7594844794 m
target node:  surface1:46:0.420261
edge index:   10
```

The coarse graph q for this edge was formally endpoint-valid (`position error 1.026e-9 m`, `normal error 0 deg`). However, after the preceding branch rescues and rebase, the old graph q could not be reused as an endpoint candidate for the actual support pose. The standard and expanded branch-rescue searches returned zero candidates for this re-based task:

```text
standard candidate count:  0
expanded candidate count:  0
```

This is classified as:

```text
TRAJECTORY_CONNECTIVITY_FAILURE
IK_BRANCH_NOT_REPRODUCED_AFTER_REBASE
```

It is not classified as a Tower collision, central-body collision, or RRT failure. RRT was not called because no valid re-based q_goal reached the trajectory stage. The fixed RRT settings remained 1200 iterations per seed and two seeds (`20260817`, `20260818`).

## 7. What the result means

The fixed 8R design demonstrates substantial task capability:

* all four representative first-layer position+normal tasks have valid collision-free endpoint branches;
* the two-candidate endpoint graph reaches the Tower-top height tolerance;
* multiple early and mid-route branch-rescued edges are Straight-valid with millimetre-scale positive reported clearance.

It does **not** demonstrate a complete Tower climb. The highest endpoint route loses a reproducible IK branch after rebase at approximately `5.759 m` in the fine validation performed here. Consequently `SUCCESS_8R_REQUIRED` is not justified.

The final status `NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE` means no complete route was found within the bounded search actually executed: two coarse candidates per surface node, the existing graph expansion budget of 300 (133 expansions were needed to exhaust the reachable frontier), the existing standard/expanded branch-rescue budgets, and the fixed RRT settings. It is not a proof that every possible 8R contact sampling, every possible IK branch, or every future morphology is impossible.

## 8. Final comparison

| Design | Endpoint graph | Full trajectory | First decisive failure | Max endpoint height |
|---|---|---|---|---:|
| `BASELINE_8R` | Critical position workspace failure | No | `POSITION_WORKSPACE` | Step-2 region only |
| `BEST_6R` | No initial valid contact state | No | `POSITION_WORKSPACE` | 0 m |
| `OPTIMIZED_8R_YAW_PITCH_YAW_PITCH` | Reaches top tolerance | No | Re-based IK branch not reproduced at 5.759 m | 37.8549988 m |

Under the evaluated candidates, 8R is materially more capable than 6R at position, normal, collision-filtered endpoint, and contact-graph levels. However, the current evidence supports “8R required for the tested endpoint capability” rather than “8R has a completed whole-Tower route.”

## 9. Checkpoints and generated results

* `models/design_results/checkpoints/whole_tower_search_checkpoint.json`（最终 8R 双候选接触图与 fine-route checkpoint）
* `models/design_results/checkpoints/best6r_whole_tower_search_checkpoint.json`（6R 初始接触失败 checkpoint）
* `models/design_results/whole_tower_routes/finite_axis_yaw_pitch_yaw_pitch_search.json`（最终 8R 接触图与 fine-route 结构化报告）
* `models/design_results/whole_tower_routes/best6r_full_search.json`（6R 全塔验证报告）
* `models/design_results/whole_tower_routes/finite_axis_yaw_pitch_yaw_pitch/rescue_routes/`（fine-route branch-rescue 轨迹目录）
* `environment/design/whole_tower_search.py`（接触图、rebase、branch rescue 与 fine validation 实现）

The research command used for the completed endpoint graph was:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  -m environment.design.whole_tower_search \
  --design axis8 \
  --max-expansions 300 \
  --candidates-per-surface 2 \
  --resume
```

## 10. Final recommendation for this checkpoint

Do not claim a completed 6R or 8R Tower-climbing design. Mark the current bounded research status as `NO_SOLUTION_FOUND_WITHIN_SEARCH_SPACE`. If research resumes later, the next technically meaningful expansion is to preserve multiple post-rebase IK branches in the contact state itself and search alternative targets around the first unresolved edge; increasing RRT iterations is not the first remedy because the current failure occurred before a valid re-based q_goal was available.
