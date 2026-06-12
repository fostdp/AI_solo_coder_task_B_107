import uuid
import logging
from datetime import datetime
from multiprocessing import Process
from fastapi import APIRouter, HTTPException, status
from typing import Dict, List, Optional

from .schemas import (
    CavePriorityRankingRequest,
    RankingTaskResponse,
    RankingResultResponse,
)
from .worker import run_nsga2, get_task_results

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/priority-nsga2", tags=["Priority NSGA-II"])

tasks: Dict = get_task_results()
_processes: Dict[str, Process] = {}
_active_processes = _processes


@router.post(
    "/rank",
    response_model=RankingTaskResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def submit_ranking_task(request: CavePriorityRankingRequest):
    task_id = str(uuid.uuid4())
    task_results = get_task_results()

    caves_data = [c.model_dump() for c in request.caves]
    params = {
        "weights": request.weights,
        "use_nsga_ii": request.use_nsga_ii,
    }

    task_results[task_id] = {
        "status": "pending",
        "results": None,
        "error": None,
        "created_at": datetime.utcnow().isoformat(),
    }

    process = Process(
        target=run_nsga2,
        args=(task_id, caves_data, params, task_results),
        daemon=True,
    )
    process.start()
    _processes[task_id] = process

    logger.info(f"已提交NSGA-II排序任务，任务ID: {task_id}，窟室数量: {len(request.caves)}")

    return RankingTaskResponse(
        task_id=task_id,
        status="pending",
        message="排序任务已提交，正在后台处理中",
    )


@router.get(
    "/result/{task_id}",
    response_model=RankingResultResponse,
)
async def get_ranking_result(task_id: str):
    task_results = get_task_results()

    if task_id not in task_results:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"任务ID {task_id} 不存在",
        )

    task_data = task_results[task_id]
    status_value = task_data.get("status", "pending")

    if status_value == "failed":
        return RankingResultResponse(
            task_id=task_id,
            status="failed",
            error=task_data.get("error", "未知错误"),
        )

    if status_value == "pending":
        return RankingResultResponse(
            task_id=task_id,
            status="pending",
        )

    results = task_data.get("results")
    if results is None:
        return RankingResultResponse(
            task_id=task_id,
            status="pending",
        )

    return RankingResultResponse(
        task_id=task_id,
        status="completed",
        results=results.get("ranking", []),
    )


@router.get("/health")
async def health_check():
    task_results = get_task_results()

    pending_count = 0
    completed_count = 0
    failed_count = 0
    active_process_count = 0

    for task_id, task_data in task_results.items():
        task_status = task_data.get("status", "pending")
        if task_status == "pending":
            pending_count += 1
        elif task_status == "completed":
            completed_count += 1
        elif task_status == "failed":
            failed_count += 1

    for task_id, process in list(_processes.items()):
        if process.is_alive():
            active_process_count += 1
        else:
            process.join(timeout=0)
            if not process.is_alive():
                _processes.pop(task_id, None)

    return {
        "status": "healthy",
        "worker": "running",
        "active_processes": active_process_count,
        "total_tasks": len(task_results),
        "pending_tasks": pending_count,
        "completed_tasks": completed_count,
        "failed_tasks": failed_count,
        "timestamp": datetime.utcnow().isoformat(),
    }
