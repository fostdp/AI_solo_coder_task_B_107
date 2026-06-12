import uuid
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any
from multiprocessing import Manager

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

from .problem_wrapper import (
    CavePriorityInput,
    CavePriorityProblem,
    compute_urgency_score,
    compute_cost_efficiency,
    compute_cultural_impact,
)

_manager = None
_task_results = None


def _get_manager():
    global _manager, _task_results
    if _manager is None:
        _manager = Manager()
        _task_results = _manager.dict()
    return _manager, _task_results


def get_task_results():
    _, task_results = _get_manager()
    return task_results


def _init_worker_process():
    global _manager, _task_results
    _manager = None
    _task_results = None


def run_nsga2(
    task_id: str,
    caves_data: List[Dict[str, Any]],
    params: Optional[Dict[str, Any]] = None,
    task_results: Optional[Dict] = None,
):
    if task_results is None:
        _, task_results = _get_manager()

    try:
        created_at = task_results.get(task_id, {}).get("created_at", datetime.utcnow().isoformat())
        task_results[task_id] = {
            "status": "pending",
            "results": None,
            "error": None,
            "created_at": created_at,
        }

        caves = [CavePriorityInput(**c) for c in caves_data]

        problem = CavePriorityProblem(caves)

        algorithm = NSGA2(
            pop_size=50,
            n_offsprings=50,
            eliminate_duplicates=True,
        )

        termination = get_termination("n_gen", 60)

        result = minimize(
            problem,
            algorithm,
            termination,
            seed=42,
            verbose=False,
        )

        pareto_F = -result.F
        pareto_X = result.X

        best_idx = 0
        best_agg = -np.inf
        for i in range(len(pareto_X)):
            weights = np.abs(pareto_X[i]) / np.sum(np.abs(pareto_X[i]))
            agg = (
                weights[0] * pareto_F[i, 0]
                + weights[1] * pareto_F[i, 1]
                + weights[2] * pareto_F[i, 2]
            )
            if agg > best_agg:
                best_agg = agg
                best_idx = i

        optimal_weights = np.abs(pareto_X[best_idx]) / np.sum(np.abs(pareto_X[best_idx]))
        weights_dict = {
            "urgency": float(optimal_weights[0]),
            "cost_efficiency": float(optimal_weights[1]),
            "cultural_impact": float(optimal_weights[2]),
        }

        ranking_results = []
        for cave in caves:
            urgency = compute_urgency_score(cave)
            efficiency = compute_cost_efficiency(cave)
            impact = compute_cultural_impact(cave)

            aggregated_score = (
                urgency * weights_dict["urgency"]
                + efficiency * weights_dict["cost_efficiency"]
                + impact * weights_dict["cultural_impact"]
            )

            ranking_results.append({
                "cave_id": cave.cave_id,
                "cave_name": cave.cave_name,
                "urgency_score": round(urgency, 4),
                "cost_efficiency_score": round(efficiency, 4),
                "cultural_impact_score": round(impact, 4),
                "aggregated_score": round(aggregated_score * 100, 4),
                "weights_used": {k: round(v, 4) for k, v in weights_dict.items()},
                "method": "pymoo_nsga_ii_multi_objective",
                "pareto_rank": 0,
                "crowding_distance": None,
                "pareto_front_size": len(pareto_F),
                "nsga_ii_summary": {
                    "population_size": 50,
                    "generations": 60,
                    "pareto_front_size": len(pareto_F),
                    "convergence_indicator": round(float(np.mean(pareto_F[:, 0])), 4),
                    "library": "pymoo",
                },
                "detail_metrics": {
                    "total_delamination_area_sqm": cave.total_delamination_area_sqm,
                    "max_severity_score": cave.max_severity_score,
                    "historical_repair_count": cave.historical_repair_count,
                    "last_repair_years_ago": cave.last_repair_years_ago,
                    "avg_visitor_flow_daily": cave.avg_visitor_flow_daily,
                    "peak_visitor_flow_daily": cave.peak_visitor_flow_daily,
                    "cultural_significance_score": cave.cultural_significance_score,
                    "structural_importance": cave.structural_importance,
                    "avg_bond_strength_remaining_mpa": cave.avg_bond_strength_remaining_mpa,
                    "active_alerts_count": cave.active_alerts_count,
                },
            })

        ranking_results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        for i, res in enumerate(ranking_results):
            res["priority_rank"] = i + 1

        final_result = {
            "ranking": ranking_results,
            "method": "pymoo_nsga_ii_multi_objective",
            "total_caves": len(caves),
            "timestamp": datetime.utcnow().isoformat(),
            "pareto_front_size": len(pareto_F),
            "pareto_front": {
                "objectives": pareto_F.tolist(),
                "weights": pareto_X.tolist(),
                "optimal_weights": weights_dict,
            },
        }

        task_results[task_id] = {
            "status": "completed",
            "results": final_result,
            "error": None,
            "created_at": created_at,
            "completed_at": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        if task_results is None:
            _, task_results = _get_manager()
        if task_id in task_results:
            created_at = task_results[task_id].get("created_at", datetime.utcnow().isoformat())
        else:
            created_at = datetime.utcnow().isoformat()

        task_results[task_id] = {
            "status": "failed",
            "results": None,
            "error": str(e),
            "created_at": created_at,
            "failed_at": datetime.utcnow().isoformat(),
        }
