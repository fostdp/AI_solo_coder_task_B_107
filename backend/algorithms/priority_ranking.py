import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta


@dataclass
class CaveConditionData:
    cave_id: str
    cave_name: str
    total_delamination_area_sqm: float
    max_severity_score: float
    historical_repair_count: int
    last_repair_years_ago: float
    avg_visitor_flow_daily: float
    peak_visitor_flow_daily: float
    cultural_significance_score: float
    structural_importance: float
    avg_bond_strength_remaining_mpa: float
    active_alerts_count: int
    wall_surfaces_count: int = 5


@dataclass
class ProtectionPriorityObjective:
    urgency_score: float
    cost_efficiency: float
    cultural_impact: float


@dataclass
class ParetoFrontSolution:
    cave_id: str
    cave_name: str
    objectives: ProtectionPriorityObjective
    rank: int
    crowding_distance: float
    weights: Dict[str, float]
    aggregated_score: float


class NSGAIISolver:
    def __init__(
        self,
        population_size: int = 50,
        max_generations: int = 100,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        eta_c: float = 20.0,
        eta_m: float = 20.0,
        random_seed: Optional[int] = 42,
        use_adaptive_mutation: bool = True,
        mutation_prob_min: float = 0.02,
        mutation_prob_max: float = 0.30,
        eta_m_min: float = 10.0,
        eta_m_max: float = 40.0,
        diversity_window: int = 5,
    ):
        self.pop_size = population_size
        self.max_gen = max_generations
        self.p_c = crossover_prob
        self.p_m_init = mutation_prob
        self.p_m = mutation_prob
        self.eta_c = eta_c
        self.eta_m_init = eta_m
        self.eta_m = eta_m
        self.use_adaptive_mutation = use_adaptive_mutation
        self.p_m_min = mutation_prob_min
        self.p_m_max = mutation_prob_max
        self.eta_m_min = eta_m_min
        self.eta_m_max = eta_m_max
        self.diversity_window = diversity_window

        self._hv_history = []
        self._pareto_size_history = []

        if random_seed is not None:
            np.random.seed(random_seed)

    def _population_spread_metric(self, objectives: np.ndarray) -> float:
        n_obj = objectives.shape[1]
        norm_obj = self._normalize_objectives(objectives)
        spreads = []
        for k in range(n_obj):
            col = norm_obj[:, k]
            spreads.append(float(np.std(col)))
        return float(np.mean(spreads)) if len(spreads) > 0 else 0.0

    def _compute_spacing(self, pareto_objectives: np.ndarray) -> float:
        n = pareto_objectives.shape[0]
        if n < 2:
            return 0.0
        norm_obj = self._normalize_objectives(pareto_objectives)
        distances = []
        for i in range(n):
            min_dist = np.inf
            for j in range(n):
                if i == j:
                    continue
                d = np.linalg.norm(norm_obj[i] - norm_obj[j])
                if d < min_dist:
                    min_dist = d
            if min_dist < np.inf:
                distances.append(min_dist)
        if len(distances) < 2:
            return 0.0
        return float(np.std(distances))

    def _adaptive_mutation_parameters(
        self,
        generation: int,
        pareto_objectives: np.ndarray,
    ) -> Tuple[float, float]:
        if not self.use_adaptive_mutation:
            return self.p_m_init, self.eta_m_init

        gen_progress = generation / max(self.max_gen - 1, 1)

        diversity = self._population_spread_metric(pareto_objectives)
        diversity = min(max(diversity, 0.0), 1.0)

        self._pareto_size_history.append(len(pareto_objectives))
        if len(self._pareto_size_history) > self.diversity_window:
            self._pareto_size_history = self._pareto_size_history[-self.diversity_window:]
        if len(self._pareto_size_history) >= 2:
            recent_growth = (
                self._pareto_size_history[-1] - self._pareto_size_history[0]
            ) / max(self._pareto_size_history[0], 1)
        else:
            recent_growth = 0.0

        stagnation_factor = 1.0
        if len(self._pareto_size_history) >= self.diversity_window and recent_growth < 0.02:
            stagnation_factor = 1.5

        diversity_factor = 1.0 - 0.6 * diversity
        stage_factor = 1.0 - 0.7 * gen_progress

        p_m_relative = min(max(
            stage_factor * diversity_factor * stagnation_factor,
            0.15,
        ), 1.5)

        current_p_m = self.p_m_init * p_m_relative
        current_p_m = min(max(current_p_m, self.p_m_min), self.p_m_max)

        eta_m_relative = 1.0 + 0.8 * gen_progress - 0.4 * diversity_factor
        current_eta_m = self.eta_m_init * eta_m_relative
        current_eta_m = min(max(current_eta_m, self.eta_m_min), self.eta_m_max)

        return float(current_p_m), float(current_eta_m)

    def _normalize_objectives(
        self,
        objectives: np.ndarray,
    ) -> np.ndarray:
        n_obj = objectives.shape[1]
        normalized = np.zeros_like(objectives, dtype=np.float64)
        
        for i in range(n_obj):
            col = objectives[:, i]
            col_min, col_max = np.min(col), np.max(col)
            if col_max > col_min:
                normalized[:, i] = (col - col_min) / (col_max - col_min)
            else:
                normalized[:, i] = 0.5
        
        return normalized

    def _dominates(
        self,
        a: np.ndarray,
        b: np.ndarray,
        maximize: Optional[List[bool]] = None,
    ) -> bool:
        if maximize is None:
            maximize = [True] * len(a)
        
        better = False
        worse = False
        
        for i in range(len(a)):
            if maximize[i]:
                if a[i] > b[i]:
                    better = True
                elif a[i] < b[i]:
                    worse = True
            else:
                if a[i] < b[i]:
                    better = True
                elif a[i] > b[i]:
                    worse = True
        
        return better and not worse

    def _fast_non_dominated_sort(
        self,
        objectives: np.ndarray,
        maximize: Optional[List[bool]] = None,
    ) -> Tuple[List[List[int]], np.ndarray]:
        n = objectives.shape[0]
        
        S = [[] for _ in range(n)]
        n_p = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        fronts = [[]]
        
        for p in range(n):
            for q in range(n):
                if p != q:
                    if self._dominates(objectives[p], objectives[q], maximize):
                        S[p].append(q)
                    elif self._dominates(objectives[q], objectives[p], maximize):
                        n_p[p] += 1
            
            if n_p[p] == 0:
                ranks[p] = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in S[p]:
                    n_p[q] -= 1
                    if n_p[q] == 0:
                        ranks[q] = i + 1
                        next_front.append(q)
            i += 1
            fronts.append(next_front)
        
        fronts.pop()
        return fronts, ranks

    def _crowding_distance(
        self,
        objectives: np.ndarray,
        front: List[int],
    ) -> np.ndarray:
        n = len(front)
        distances = np.zeros(n)
        
        if n <= 2:
            distances[:] = np.inf
            return distances
        
        n_obj = objectives.shape[1]
        
        for m in range(n_obj):
            sorted_idx = sorted(range(n), key=lambda i: objectives[front[i], m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            min_val = objectives[front[sorted_idx[0]], m]
            max_val = objectives[front[sorted_idx[-1]], m]
            obj_range = max_val[0] - min_val[0] if isinstance(min_val, tuple) else max_val - min_val
            
            if obj_range > 0:
                for i in range(1, n - 1):
                    prev_val = objectives[front[sorted_idx[i - 1]], m]
                    next_val = objectives[front[sorted_idx[i + 1]], m]
                    distances[sorted_idx[i]] += (next_val - prev_val) / obj_range
        
        return distances

    def _tournament_selection(
        self,
        population: np.ndarray,
        ranks: np.ndarray,
        crowding_distances: np.ndarray,
        tournament_size: int = 2,
    ) -> np.ndarray:
        n = population.shape[0]
        selected = np.zeros_like(population)
        
        for i in range(n):
            candidates = np.random.choice(n, tournament_size, replace=False)
            best_idx = candidates[0]
            
            for idx in candidates[1:]:
                if ranks[idx] < ranks[best_idx]:
                    best_idx = idx
                elif ranks[idx] == ranks[best_idx]:
                    if crowding_distances[idx] > crowding_distances[best_idx]:
                        best_idx = idx
            
            selected[i] = population[best_idx].copy()
        
        return selected

    def _sbx_crossover(
        self,
        parent1: np.ndarray,
        parent2: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n_var = len(parent1)
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(n_var):
            if np.random.rand() > self.p_c:
                continue
            
            if np.random.rand() < 0.5:
                y1 = min(parent1[i], parent2[i])
                y2 = max(parent1[i], parent2[i])
                
                if y2 > y1:
                    rand = np.random.rand()
                    beta = 1.0 + (2.0 * (y1 - lower_bounds[i]) / (y2 - y1))
                    alpha = 2.0 - (beta ** -(self.eta_c + 1.0))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    
                    child1[i] = 0.5 * ((y1 + y2) - beta_q * (y2 - y1))
                    
                    beta = 1.0 + (2.0 * (upper_bounds[i] - y2) / (y2 - y1))
                    alpha = 2.0 - (beta ** -(self.eta_c + 1.0))
                    
                    if rand <= 1.0 / alpha:
                        beta_q = (rand * alpha) ** (1.0 / (self.eta_c + 1.0))
                    else:
                        beta_q = (1.0 / (2.0 - rand * alpha)) ** (1.0 / (self.eta_c + 1.0))
                    
                    child2[i] = 0.5 * ((y1 + y2) + beta_q * (y2 - y1))
                    
                    child1[i] = np.clip(child1[i], lower_bounds[i], upper_bounds[i])
                    child2[i] = np.clip(child2[i], lower_bounds[i], upper_bounds[i])
        
        return child1, child2

    def _polynomial_mutation(
        self,
        individual: np.ndarray,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        p_m: Optional[float] = None,
        eta_m: Optional[float] = None,
    ) -> np.ndarray:
        n_var = len(individual)
        mutant = individual.copy()
        cur_pm = self.p_m if p_m is None else p_m
        cur_etam = self.eta_m if eta_m is None else eta_m

        for i in range(n_var):
            if np.random.rand() > cur_pm:
                continue

            y = mutant[i]
            y_low = lower_bounds[i]
            y_high = upper_bounds[i]

            if y_high > y_low:
                delta1 = (y - y_low) / (y_high - y_low)
                delta2 = (y_high - y) / (y_high - y_low)

                rand = np.random.rand()
                mut_pow = 1.0 / (cur_etam + 1.0)

                if rand <= 0.5:
                    xy = 1.0 - delta1
                    val = 2.0 * rand + (1.0 - 2.0 * rand) * (xy ** (cur_etam + 1.0))
                    delta_q = val ** mut_pow - 1.0
                else:
                    xy = 1.0 - delta2
                    val = 2.0 * (1.0 - rand) + 2.0 * (rand - 0.5) * (xy ** (cur_etam + 1.0))
                    delta_q = 1.0 - val ** mut_pow

                mutant[i] = y + delta_q * (y_high - y_low)
                mutant[i] = np.clip(mutant[i], y_low, y_high)

        return mutant

    def solve(
        self,
        n_variables: int,
        n_objectives: int,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
        objective_function,
        maximize: Optional[List[bool]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if maximize is None:
            maximize = [True] * n_objectives
        
        population = np.random.uniform(
            low=lower_bounds,
            high=upper_bounds,
            size=(self.pop_size, n_variables),
        )
        
        pop_objectives = np.array([
            objective_function(ind) for ind in population
        ])
        
        norm_objectives = self._normalize_objectives(pop_objectives)
        
        for gen in range(self.max_gen):
            fronts, ranks = self._fast_non_dominated_sort(norm_objectives, maximize)

            current_pareto = pop_objectives[fronts[0]] if len(fronts) > 0 else pop_objectives
            self.p_m, self.eta_m = self._adaptive_mutation_parameters(gen, current_pareto)

            crowding = np.zeros(self.pop_size)
            for front in fronts:
                cd = self._crowding_distance(norm_objectives, front)
                for idx, f_idx in enumerate(front):
                    crowding[f_idx] = cd[idx]

            parents = self._tournament_selection(population, ranks, crowding)

            offspring = np.zeros_like(population)
            for i in range(0, self.pop_size, 2):
                p1_idx = i
                p2_idx = i + 1 if i + 1 < self.pop_size else 0

                child1, child2 = self._sbx_crossover(
                    parents[p1_idx], parents[p2_idx],
                    lower_bounds, upper_bounds,
                )
                child1 = self._polynomial_mutation(
                    child1, lower_bounds, upper_bounds,
                    p_m=self.p_m, eta_m=self.eta_m,
                )
                child2 = self._polynomial_mutation(
                    child2, lower_bounds, upper_bounds,
                    p_m=self.p_m, eta_m=self.eta_m,
                )

                offspring[i] = child1
                if i + 1 < self.pop_size:
                    offspring[i + 1] = child2

            combined_pop = np.vstack([population, offspring])
            combined_obj = np.vstack([pop_objectives, np.array([
                objective_function(ind) for ind in offspring
            ])])
            combined_norm = self._normalize_objectives(combined_obj)

            fronts, ranks = self._fast_non_dominated_sort(combined_norm, maximize)

            new_population = []
            new_objectives = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                    new_objectives.extend(front)
                else:
                    remaining = self.pop_size - len(new_population)
                    if remaining > 0:
                        cd = self._crowding_distance(combined_norm, front)
                        sorted_by_cd = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                        selected = [front[i] for i in sorted_by_cd[:remaining]]
                        new_population.extend(selected)
                        new_objectives.extend(selected)
                    break

            population = combined_pop[new_population]
            pop_objectives = combined_obj[new_population]
            norm_objectives = self._normalize_objectives(pop_objectives)

        final_fronts, final_ranks = self._fast_non_dominated_sort(
            self._normalize_objectives(pop_objectives), maximize
        )
        pareto_front = pop_objectives[final_fronts[0]] if len(final_fronts) > 0 else pop_objectives
        pareto_solutions = population[final_fronts[0]] if len(final_fronts) > 0 else population

        final_spacing = self._compute_spacing(pareto_front)
        final_diversity = self._population_spread_metric(pop_objectives)

        return {
            "population": population,
            "objectives": pop_objectives,
            "pareto_front": pareto_front,
            "pareto_solutions": pareto_solutions,
            "fronts": final_fronts,
            "ranks": final_ranks,
            "generations": self.max_gen,
            "population_size": self.pop_size,
            "n_objectives": n_objectives,
            "pareto_front_size": len(pareto_front),
            "adaptive_mutation_used": self.use_adaptive_mutation,
            "final_mutation_prob": round(float(self.p_m), 4),
            "final_mutation_eta_m": round(float(self.eta_m), 4),
            "pareto_spacing": round(float(final_spacing), 6),
            "population_diversity": round(float(final_diversity), 6),
        }


class CaveProtectionPriorityRanking:
    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
    ):
        if weights is None:
            weights = {
                "urgency": 0.50,
                "cost_efficiency": 0.25,
                "cultural_impact": 0.25,
            }
        
        self.weights = weights
        self._validate_weights()

    def _validate_weights(self):
        total = sum(self.weights.values())
        if abs(total - 1.0) > 0.001:
            self.weights = {k: v / total for k, v in self.weights.items()}

    def compute_urgency_score(
        self,
        cave: CaveConditionData,
    ) -> float:
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

    def compute_cost_efficiency(
        self,
        cave: CaveConditionData,
    ) -> float:
        area_per_wall = cave.total_delamination_area_sqm / max(cave.wall_surfaces_count, 1)
        repair_frequency = cave.historical_repair_count / max(cave.last_repair_years_ago + 0.1, 0.1)
        
        base_cost = 50000 + area_per_wall * 8000 + cave.historical_repair_count * 2000
        expected_repair_interval = 5.0 + cave.avg_bond_strength_remaining_mpa * 10.0
        
        efficiency = expected_repair_interval / max(base_cost / 100000.0, 1.0)
        normalized_efficiency = min(efficiency / 5.0, 1.0)
        
        return float(normalized_efficiency)

    def compute_cultural_impact(
        self,
        cave: CaveConditionData,
    ) -> float:
        visitor_impact = (cave.avg_visitor_flow_daily + cave.peak_visitor_flow_daily * 0.5) / 10000.0
        visitor_impact = min(visitor_impact, 1.0)
        
        impact = (
            cave.cultural_significance_score * 0.45
            + cave.structural_importance * 0.30
            + visitor_impact * 0.25
        )
        
        return float(min(max(impact, 0.0), 1.0))

    def _objective_function_factory(
        self,
        caves: List[CaveConditionData],
    ):
        n_caves = len(caves)
        
        def obj_func(weights: np.ndarray) -> np.ndarray:
            norm_weights = np.abs(weights) / np.sum(np.abs(weights))
            
            urgency_scores = np.array([self.compute_urgency_score(c) for c in caves])
            efficiency_scores = np.array([self.compute_cost_efficiency(c) for c in caves])
            impact_scores = np.array([self.compute_cultural_impact(c) for c in caves])
            
            weighted_urgency = float(np.sum(norm_weights[0] * urgency_scores))
            weighted_efficiency = float(np.sum(norm_weights[1] * efficiency_scores))
            weighted_impact = float(np.sum(norm_weights[2] * impact_scores))
            
            return np.array([weighted_urgency, weighted_efficiency, weighted_impact])
        
        return obj_func

    def rank_caves(
        self,
        caves: List[CaveConditionData],
        use_nsga_ii: bool = True,
    ) -> List[Dict]:
        if use_nsga_ii and len(caves) >= 2:
            return self._rank_with_nsga_ii(caves)
        else:
            return self._rank_with_simple_weighting(caves)

    def _rank_with_simple_weighting(
        self,
        caves: List[CaveConditionData],
    ) -> List[Dict]:
        results = []
        for cave in caves:
            urgency = self.compute_urgency_score(cave)
            efficiency = self.compute_cost_efficiency(cave)
            impact = self.compute_cultural_impact(cave)
            
            aggregated_score = (
                urgency * self.weights["urgency"]
                + efficiency * self.weights["cost_efficiency"]
                + impact * self.weights["cultural_impact"]
            )
            
            results.append({
                "cave_id": cave.cave_id,
                "cave_name": cave.cave_name,
                "urgency_score": round(urgency, 4),
                "cost_efficiency_score": round(efficiency, 4),
                "cultural_impact_score": round(impact, 4),
                "aggregated_score": round(aggregated_score, 4),
                "weights_used": self.weights,
                "method": "simple_weighted_sum",
                "pareto_rank": None,
                "crowding_distance": None,
                "detail_metrics": {
                    "total_delamination_area_sqm": cave.total_delamination_area_sqm,
                    "max_severity_score": cave.max_severity_score,
                    "historical_repair_count": cave.historical_repair_count,
                    "last_repair_years_ago": cave.last_repair_years_ago,
                    "avg_visitor_flow_daily": cave.avg_visitor_flow_daily,
                    "avg_bond_strength_remaining_mpa": cave.avg_bond_strength_remaining_mpa,
                    "active_alerts_count": cave.active_alerts_count,
                },
            })
        
        results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        for i, res in enumerate(results):
            res["priority_rank"] = i + 1
        
        return results

    def _rank_with_nsga_ii(
        self,
        caves: List[CaveConditionData],
    ) -> List[Dict]:
        obj_func = self._objective_function_factory(caves)
        
        lower_bounds = np.array([0.1, 0.1, 0.1])
        upper_bounds = np.array([1.0, 1.0, 1.0])
        
        solver = NSGAIISolver(
            population_size=40,
            max_generations=60,
            crossover_prob=0.85,
            mutation_prob=0.15,
        )
        
        solve_result = solver.solve(
            n_variables=3,
            n_objectives=3,
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            objective_function=obj_func,
            maximize=[True, True, True],
        )

        final_pop = solve_result["population"]
        final_obj = solve_result["objectives"]
        fronts = solve_result["fronts"]
        ranks = solve_result["ranks"]

        norm_obj = solver._normalize_objectives(final_obj)
        crowding = np.zeros(len(final_obj))
        for front in fronts:
            cd = solver._crowding_distance(norm_obj, front)
            for idx, f_idx in enumerate(front):
                crowding[f_idx] = cd[idx]

        best_idx = 0
        best_agg = -np.inf
        for i in range(len(final_pop)):
            weights = np.abs(final_pop[i]) / np.sum(np.abs(final_pop[i]))
            agg = weights[0] * final_obj[i, 0] + weights[1] * final_obj[i, 1] + weights[2] * final_obj[i, 2]
            if agg > best_agg:
                best_agg = agg
                best_idx = i

        optimal_weights = np.abs(final_pop[best_idx]) / np.sum(np.abs(final_pop[best_idx]))
        self.weights = {
            "urgency": float(optimal_weights[0]),
            "cost_efficiency": float(optimal_weights[1]),
            "cultural_impact": float(optimal_weights[2]),
        }

        results = []
        for cave in caves:
            urgency = self.compute_urgency_score(cave)
            efficiency = self.compute_cost_efficiency(cave)
            impact = self.compute_cultural_impact(cave)

            aggregated_score = (
                urgency * self.weights["urgency"]
                + efficiency * self.weights["cost_efficiency"]
                + impact * self.weights["cultural_impact"]
            )

            results.append({
                "cave_id": cave.cave_id,
                "cave_name": cave.cave_name,
                "urgency_score": round(urgency, 4),
                "cost_efficiency_score": round(efficiency, 4),
                "cultural_impact_score": round(impact, 4),
                "aggregated_score": round(aggregated_score * 100, 4),
                "weights_used": {k: round(v, 4) for k, v in self.weights.items()},
                "method": "nsga_ii_multi_objective",
                "pareto_rank": int(ranks[best_idx]) if best_idx < len(ranks) else 0,
                "crowding_distance": round(float(crowding[best_idx]), 4) if best_idx < len(crowding) and not np.isinf(crowding[best_idx]) else None,
                "pareto_front_size": solve_result["pareto_front_size"],
                "nsga_ii_summary": {
                    "population_size": 40,
                    "generations": 60,
                    "pareto_front_size": solve_result["pareto_front_size"],
                    "convergence_indicator": round(float(np.mean(final_obj[:, 0])), 4),
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
        
        results.sort(key=lambda x: x["aggregated_score"], reverse=True)
        for i, res in enumerate(results):
            res["priority_rank"] = i + 1
        
        return results


def generate_simulated_visitor_flow(
    cave_id: str,
    days: int = 365,
    base_daily: float = 500.0,
) -> Dict:
    dates = []
    visitor_counts = []
    
    for d in range(days):
        date = datetime(2025, 1, 1) + timedelta(days=d)
        is_weekend = date.weekday() >= 5
        is_peak_season = 5 <= date.month <= 10
        is_holiday = (date.month == 10 and 1 <= date.day <= 7) or (date.month == 5 and 1 <= date.day <= 5)
        
        multiplier = 1.0
        if is_weekend:
            multiplier *= 1.5
        if is_peak_season:
            multiplier *= 2.0
        if is_holiday:
            multiplier *= 3.0
        
        noise = np.random.normal(1.0, 0.15)
        count = int(base_daily * multiplier * noise)
        
        dates.append(date.isoformat())
        visitor_counts.append(max(count, 0))
    
    return {
        "cave_id": cave_id,
        "dates": dates,
        "daily_visitors": visitor_counts,
        "avg_daily": round(float(np.mean(visitor_counts)), 2),
        "peak_daily": int(np.max(visitor_counts)),
        "min_daily": int(np.min(visitor_counts)),
        "total_annual": int(np.sum(visitor_counts)),
        "peak_season_avg": round(float(np.mean([v for i, v in enumerate(visitor_counts) if 5 <= (datetime.fromisoformat(dates[i]).month) <= 10])), 2),
        "off_peak_avg": round(float(np.mean([v for i, v in enumerate(visitor_counts) if not (5 <= (datetime.fromisoformat(dates[i]).month) <= 10)])), 2),
    }
