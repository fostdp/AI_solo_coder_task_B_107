import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import numpy as np
from apps.priority_nsga2.problem_wrapper import (
    CavePriorityInput,
    CavePriorityProblem,
    compute_urgency_score,
    compute_cost_efficiency,
    compute_cultural_impact,
)
from apps.priority_nsga2.worker import run_nsga2


def make_mock_caves():
    return [
        CavePriorityInput(
            cave_id="c001",
            cave_name="第1窟",
            total_delamination_area_sqm=3.5,
            max_severity_score=85.0,
            historical_repair_count=4,
            last_repair_years_ago=8.0,
            avg_visitor_flow_daily=1200.0,
            peak_visitor_flow_daily=3500.0,
            cultural_significance_score=0.9,
            structural_importance=0.85,
            avg_bond_strength_remaining_mpa=0.3,
            active_alerts_count=3,
            wall_surfaces_count=5,
        ),
        CavePriorityInput(
            cave_id="c002",
            cave_name="第2窟",
            total_delamination_area_sqm=1.0,
            max_severity_score=40.0,
            historical_repair_count=1,
            last_repair_years_ago=2.0,
            avg_visitor_flow_daily=500.0,
            peak_visitor_flow_daily=1500.0,
            cultural_significance_score=0.5,
            structural_importance=0.4,
            avg_bond_strength_remaining_mpa=0.7,
            active_alerts_count=0,
            wall_surfaces_count=4,
        ),
        CavePriorityInput(
            cave_id="c003",
            cave_name="第3窟",
            total_delamination_area_sqm=4.8,
            max_severity_score=95.0,
            historical_repair_count=8,
            last_repair_years_ago=15.0,
            avg_visitor_flow_daily=2000.0,
            peak_visitor_flow_daily=6000.0,
            cultural_significance_score=0.95,
            structural_importance=0.95,
            avg_bond_strength_remaining_mpa=0.15,
            active_alerts_count=5,
            wall_surfaces_count=6,
        ),
    ]


def test_score_functions():
    print("=" * 60)
    print("测试 1: 三个评分函数 (urgency / cost_efficiency / cultural_impact)")
    caves = make_mock_caves()
    for c in caves:
        u = compute_urgency_score(c)
        e = compute_cost_efficiency(c)
        i = compute_cultural_impact(c)
        print(f"  {c.cave_name}: urgency={u:.4f}, efficiency={e:.4f}, impact={i:.4f}")
        assert 0.0 <= u <= 1.0, f"urgency out of range for {c.cave_id}"
        assert 0.0 <= e <= 1.0, f"efficiency out of range for {c.cave_id}"
        assert 0.0 <= i <= 1.0, f"impact out of range for {c.cave_id}"
    print("  [PASS] 所有评分在 [0,1] 范围内")
    print()


def test_problem_definition():
    print("=" * 60)
    print("测试 2: CavePriorityProblem 继承与参数定义")
    caves = make_mock_caves()
    problem = CavePriorityProblem(caves)
    from pymoo.core.problem import Problem
    assert isinstance(problem, Problem), "CavePriorityProblem 不是 Problem 的子类"
    assert problem.n_var == 3, f"n_var should be 3, got {problem.n_var}"
    assert problem.n_obj == 3, f"n_obj should be 3, got {problem.n_obj}"
    assert problem.n_ieq_constr == 0, f"n_ieq_constr should be 0, got {problem.n_ieq_constr}"
    assert list(problem.xl) == [0.1, 0.1, 0.1], f"xl mismatch: {problem.xl}"
    assert list(problem.xu) == [1.0, 1.0, 1.0], f"xu mismatch: {problem.xu}"
    print(f"  n_var={problem.n_var}, n_obj={problem.n_obj}, n_ieq_constr={problem.n_ieq_constr}")
    print(f"  xl={list(problem.xl)}, xu={list(problem.xu)}")
    print("  [PASS] Problem 定义正确")
    print()


def test_evaluate_sign():
    print("=" * 60)
    print("测试 3: _evaluate 返回目标值取负 (最大化问题转最小化)")
    caves = make_mock_caves()
    problem = CavePriorityProblem(caves)
    X = np.array([
        [0.33, 0.33, 0.34],
        [0.6, 0.2, 0.2],
        [0.1, 0.5, 0.4],
    ])
    out = {}
    problem._evaluate(X, out)
    F = out["F"]
    assert F.shape == (3, 3), f"F shape should be (3,3), got {F.shape}"
    urgency_scores = np.array([compute_urgency_score(c) for c in caves])
    efficiency_scores = np.array([compute_cost_efficiency(c) for c in caves])
    impact_scores = np.array([compute_cultural_impact(c) for c in caves])

    for i in range(3):
        weights = np.abs(X[i]) / np.sum(np.abs(X[i]))
        expected_u = float(np.sum(weights[0] * urgency_scores))
        expected_e = float(np.sum(weights[1] * efficiency_scores))
        expected_i = float(np.sum(weights[2] * impact_scores))
        np.testing.assert_almost_equal(F[i, 0], -expected_u, decimal=6,
                                       err_msg=f"Row {i}: urgency sign/value wrong")
        np.testing.assert_almost_equal(F[i, 1], -expected_e, decimal=6,
                                       err_msg=f"Row {i}: efficiency sign/value wrong")
        np.testing.assert_almost_equal(F[i, 2], -expected_i, decimal=6,
                                       err_msg=f"Row {i}: impact sign/value wrong")
    print(f"  F shape: {F.shape}")
    print(f"  F (negated objectives):\n{F}")
    print("  [PASS] 所有 3 个目标都被正确取负")
    print()


def test_worker_in_memory():
    print("=" * 60)
    print("测试 4: worker.run_nsga2 核心逻辑 (非多进程，传入普通 dict)")
    caves = make_mock_caves()
    caves_data = [
        {
            "cave_id": c.cave_id,
            "cave_name": c.cave_name,
            "total_delamination_area_sqm": c.total_delamination_area_sqm,
            "max_severity_score": c.max_severity_score,
            "historical_repair_count": c.historical_repair_count,
            "last_repair_years_ago": c.last_repair_years_ago,
            "avg_visitor_flow_daily": c.avg_visitor_flow_daily,
            "peak_visitor_flow_daily": c.peak_visitor_flow_daily,
            "cultural_significance_score": c.cultural_significance_score,
            "structural_importance": c.structural_importance,
            "avg_bond_strength_remaining_mpa": c.avg_bond_strength_remaining_mpa,
            "active_alerts_count": c.active_alerts_count,
            "wall_surfaces_count": c.wall_surfaces_count,
        }
        for c in caves
    ]
    task_id = "test-task-001"
    task_results = {}
    params = {"weights": None, "use_nsga_ii": True}
    run_nsga2(task_id, caves_data, params, task_results)
    assert task_id in task_results, "task_results 中未写入 task_id"
    assert task_results[task_id]["status"] == "completed", \
        f"任务失败: {task_results[task_id].get('error')}"
    results = task_results[task_id]["results"]
    assert results is not None
    assert "ranking" in results
    assert len(results["ranking"]) == 3
    ranks = [r["priority_rank"] for r in results["ranking"]]
    assert sorted(ranks) == [1, 2, 3], f"排序 rank 不正确: {ranks}"
    scores = [r["aggregated_score"] for r in results["ranking"]]
    assert scores == sorted(scores, reverse=True), "排序分数不是降序"
    assert results["pareto_front_size"] > 0
    assert "pareto_front" in results
    print(f"  状态: {task_results[task_id]['status']}")
    print(f"  帕累托前沿大小: {results['pareto_front_size']}")
    for r in results["ranking"]:
        print(f"    Rank {r['priority_rank']}: {r['cave_name']} "
              f"score={r['aggregated_score']:.4f} "
              f"weights={r['weights_used']}")
    print("  [PASS] worker 核心逻辑运行成功，排序结果正确")
    print()


def test_task_dict_management():
    print("=" * 60)
    print("测试 5: api.py 模块 tasks 和 _processes 全局字典")
    from apps.priority_nsga2 import api
    assert hasattr(api, "tasks"), "api 模块缺少 tasks 全局字典"
    assert hasattr(api, "_processes"), "api 模块缺少 _processes 全局字典"
    assert isinstance(api.tasks, dict) or hasattr(api.tasks, "__getitem__"), \
        "tasks 不是可映射对象"
    assert isinstance(api._processes, dict), "_processes 不是 dict"
    print(f"  tasks type: {type(api.tasks).__name__}")
    print(f"  _processes type: {type(api._processes).__name__}, len={len(api._processes)}")
    print("  [PASS] api.py 全局字典存在")
    print()


def main():
    print("\n" + "=" * 60)
    print("  Priority NSGA-II 核心逻辑单元测试")
    print("=" * 60 + "\n")
    try:
        test_score_functions()
        test_problem_definition()
        test_evaluate_sign()
        test_worker_in_memory()
        test_task_dict_management()
        print("=" * 60)
        print("  全部 5 项测试通过 [ALL PASS]")
        print("=" * 60)
        return 0
    except AssertionError as e:
        print(f"\n[FAIL] 断言失败: {e}")
        return 1
    except Exception as e:
        print(f"\n[ERROR] 未预期异常: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    sys.exit(main())
