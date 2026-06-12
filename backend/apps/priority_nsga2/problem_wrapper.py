import numpy as np
from typing import List, Dict
from pymoo.core.problem import Problem


class CavePriorityInput:
    def __init__(
        self,
        cave_id: str,
        cave_name: str,
        total_delamination_area_sqm: float,
        max_severity_score: float,
        historical_repair_count: int,
        last_repair_years_ago: float,
        avg_visitor_flow_daily: float,
        peak_visitor_flow_daily: float,
        cultural_significance_score: float,
        structural_importance: float,
        avg_bond_strength_remaining_mpa: float,
        active_alerts_count: int,
        wall_surfaces_count: int = 5,
    ):
        self.cave_id = cave_id
        self.cave_name = cave_name
        self.total_delamination_area_sqm = total_delamination_area_sqm
        self.max_severity_score = max_severity_score
        self.historical_repair_count = historical_repair_count
        self.last_repair_years_ago = last_repair_years_ago
        self.avg_visitor_flow_daily = avg_visitor_flow_daily
        self.peak_visitor_flow_daily = peak_visitor_flow_daily
        self.cultural_significance_score = cultural_significance_score
        self.structural_importance = structural_importance
        self.avg_bond_strength_remaining_mpa = avg_bond_strength_remaining_mpa
        self.active_alerts_count = active_alerts_count
        self.wall_surfaces_count = wall_surfaces_count


def compute_urgency_score(cave: CavePriorityInput) -> float:
    area_factor = min(cave.total_delamination_area_sqm / 5.0, 1.0)
    severity_factor = cave.max_severity_score / 100.0
    repair_factor = min(cave.historical_repair_count / 10.0, 1.0) * 0.3
    last_repair_factor = min(cave.last_repair_years_ago / 20.0, 1.0) * 0.2
    bond_strength_factor = max(0.0, 1.0 - cave.avg_bond_strength_remaining_mpa / 0.8)
    alerts_factor = min(cave.active_alerts_count / 5.0, 1.0) * 0.3

    urgency = (
        area_factor * 0.35
        + severity_factor * 0.25
        + bond_strength_factor * 0.20
        + repair_factor
        + last_repair_factor
        + alerts_factor
    )

    return float(min(max(urgency, 0.0), 1.0))


def compute_cost_efficiency(cave: CavePriorityInput) -> float:
    area_per_wall = cave.total_delamination_area_sqm / max(cave.wall_surfaces_count, 1)
    repair_frequency = cave.historical_repair_count / max(cave.last_repair_years_ago + 0.1, 0.1)

    base_cost = 50000 + area_per_wall * 8000 + cave.historical_repair_count * 2000
    expected_repair_interval = 5.0 + cave.avg_bond_strength_remaining_mpa * 10.0

    efficiency = expected_repair_interval / max(base_cost / 100000.0, 1.0)
    normalized_efficiency = min(efficiency / 5.0, 1.0)

    return float(normalized_efficiency)


def compute_cultural_impact(cave: CavePriorityInput) -> float:
    visitor_impact = (cave.avg_visitor_flow_daily + cave.peak_visitor_flow_daily * 0.5) / 10000.0
    visitor_impact = min(visitor_impact, 1.0)

    impact = (
        cave.cultural_significance_score * 0.45
        + cave.structural_importance * 0.30
        + visitor_impact * 0.25
    )

    return float(min(max(impact, 0.0), 1.0))


class CavePriorityProblem(Problem):
    def __init__(self, caves: List[CavePriorityInput]):
        self.caves = caves
        self.urgency_scores = np.array([compute_urgency_score(c) for c in caves])
        self.efficiency_scores = np.array([compute_cost_efficiency(c) for c in caves])
        self.impact_scores = np.array([compute_cultural_impact(c) for c in caves])

        super().__init__(
            n_var=3,
            n_obj=3,
            n_ieq_constr=0,
            xl=np.array([0.1, 0.1, 0.1]),
            xu=np.array([1.0, 1.0, 1.0]),
        )

    def _evaluate(self, X, out, *args, **kwargs):
        n_individuals = X.shape[0]
        F = np.zeros((n_individuals, 3))

        for i in range(n_individuals):
            weights = np.abs(X[i]) / np.sum(np.abs(X[i]))

            weighted_urgency = float(np.sum(weights[0] * self.urgency_scores))
            weighted_efficiency = float(np.sum(weights[1] * self.efficiency_scores))
            weighted_impact = float(np.sum(weights[2] * self.impact_scores))

            F[i, 0] = -weighted_urgency
            F[i, 1] = -weighted_efficiency
            F[i, 2] = -weighted_impact

        out["F"] = F
