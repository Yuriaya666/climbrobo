
# Morphology Research Status

## Status

READY_TO_START

## Current objective

建立可复用于 symmetric 6R / 8R 的整塔 contact-state evaluator，
然后优先开展 symmetric 6R topology + link-length search。

## Fixed assumptions

- candidate link collision envelope radius: 65 mm
- central body: real mesh
- suction ends: current validated geometry
- joints/motors: independent proxies
- Tower: real Tower.STL
- morphology candidates: symmetric 6R / symmetric 8R only
- preferred solution: robust 6R if feasible

## Completed foundations

- continuous attach lines
- foot/surface decoupling
- rebase
- multi-start IK
- multi-branch position+normal IK
- collision-aware endpoint filtering
- Straight planner
- RRT-Connect
- representative optimized 8R tasks validated

## Next action

Build and validate reusable coarse-to-fine whole-tower contact-state evaluator.

## Best 6R

Not evaluated yet.

## Best 8R

Current optimized YAW-PITCH-YAW-PITCH candidate is locally validated on representative tasks, but whole-tower feasibility has not been established.

## Stop condition

Not reached.
