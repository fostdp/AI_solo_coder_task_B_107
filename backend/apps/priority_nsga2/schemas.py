from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any


class CavePriorityInput(BaseModel):
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


class CavePriorityRankingRequest(BaseModel):
    caves: List[CavePriorityInput]
    weights: Optional[Dict[str, float]] = None
    use_nsga_ii: bool = True


class CavePriorityRankingItem(BaseModel):
    cave_id: str
    cave_name: str
    priority_rank: int
    urgency_score: float
    cost_efficiency_score: float
    cultural_impact_score: float
    aggregated_score: float
    weights_used: Dict[str, float]
    method: str
    pareto_rank: Optional[int] = None
    crowding_distance: Optional[float] = None
    nsga_ii_summary: Optional[Dict] = None
    detail_metrics: Dict


class CavePriorityRankingResponse(BaseModel):
    ranking: List[CavePriorityRankingItem]
    method: str
    total_caves: int
    timestamp: str
    pareto_front_size: Optional[int] = None


class RankingTaskResponse(BaseModel):
    task_id: str
    status: str
    message: str


class RankingResultResponse(BaseModel):
    task_id: str
    status: str
    results: Optional[List[Dict]] = None
    error: Optional[str] = None
