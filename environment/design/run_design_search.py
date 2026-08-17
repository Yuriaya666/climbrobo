"""运行任务驱动的6R/8R机构设计搜索并保存checkpoint。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from environment.design.length_optimizer import LengthSearchConfig, optimize_lengths
from environment.design.morphology import BaselineGeometry
from environment.design.task_suite import build_task_suite, summarize_task_suite
from environment.design.topology_search import enumerate_symmetric_6r
from environment.paths import ProjectPaths


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="6R/8R任务驱动机构设计搜索")
    parser.add_argument("--repo-root", type=Path, default=None)
    parser.add_argument("--maxiter", type=int, default=5)
    parser.add_argument("--popsize", type=int, default=5)
    parser.add_argument("--target-limit", type=int, default=1)
    parser.add_argument("--task-targets", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260815)
    parser.add_argument("--full-validation", action="store_true")
    return parser


def _write_checkpoint(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_summary(path: Path, results) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [result.to_json_dict() for result in results]
    fields = [
        "name", "topology_id", "dof", "per_side_dof", "remove_pair_index",
        "link_lengths_m", "task_success_count", "task_count", "position_success_count",
        "worst_position_error_m", "worst_normal_error_deg", "runtime_s", "de_nfev",
        "de_nit", "boundary_hits", "task_failure_types",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def main() -> None:
    args = build_parser().parse_args()
    paths = ProjectPaths.from_repo_root(args.repo_root)
    result_dir = paths.repo_root / "models" / "design_results"
    checkpoint_dir = result_dir / "checkpoints"
    result_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_task_suite(paths, max_targets_per_task=args.task_targets)
    (result_dir / "task_suite_results.csv").write_text("", encoding="utf-8")
    with (result_dir / "task_suite_results.csv").open("w", encoding="utf-8", newline="") as handle:
        rows = summarize_task_suite(tasks)
        fields = sorted({key for row in rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    config = LengthSearchConfig(
        seed=args.seed,
        maxiter=args.maxiter,
        popsize=args.popsize,
        target_limit=args.target_limit,
    )
    geometry = BaselineGeometry.from_project(paths)
    results = []

    baseline = optimize_lengths(
        geometry,
        tasks,
        kind="baseline_8r",
        remove_pair_index=None,
        config=LengthSearchConfig(
            seed=args.seed,
            maxiter=0,
            popsize=1,
            target_limit=args.target_limit,
        ),
        output_dir=result_dir,
    )
    results.append(baseline)
    (result_dir / "baseline_8r.json").write_text(
        json.dumps(baseline.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_checkpoint(checkpoint_dir / "baseline_done.json", baseline.to_json_dict())
    print(f"BASELINE_8R: position {baseline.position_success_count}/{baseline.task_count}, worst={baseline.worst_position_error_m:.6f} m")

    six = enumerate_symmetric_6r(
        geometry,
        tasks,
        config=config,
        output_dir=result_dir,
    )
    results.extend(six.candidates)
    _write_checkpoint(
        checkpoint_dir / "6r_coarse_search_done.json",
        {"candidates": [result.to_json_dict() for result in six.candidates]},
    )
    print(f"BEST_6R coarse: {six.best.name}, position {six.best.position_success_count}/{six.best.task_count}, worst={six.best.worst_position_error_m:.6f} m")

    best8 = optimize_lengths(
        geometry,
        tasks,
        kind="optimized_8r",
        remove_pair_index=None,
        config=config,
        output_dir=result_dir,
    )
    results.append(best8)
    _write_checkpoint(checkpoint_dir / "8r_optimization_done.json", best8.to_json_dict())
    (result_dir / "best_8r.json").write_text(
        json.dumps(best8.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (result_dir / "best_6r.json").write_text(
        json.dumps(six.best.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_summary(result_dir / "design_search_summary.csv", results)
    print(f"BEST_8R: position {best8.position_success_count}/{best8.task_count}, worst={best8.worst_position_error_m:.6f} m")
    print(f"结果目录：{result_dir}（机构设计搜索结果与checkpoint）")


if __name__ == "__main__":
    main()

