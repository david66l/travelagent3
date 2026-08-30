"""FastAPI entrypoint for the standalone VRP solver service."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, status

from vrp_solver_service.models import SolverRequest, SolverResponse
from vrp_solver_service.solver import TravelVRPSolver

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TravelAgent VRP Solver Service",
    description="Independent CP-SAT based itinerary planning microservice.",
    version="0.5.0",
)

_solver = TravelVRPSolver()


@app.get("/health", status_code=status.HTTP_200_OK)
async def health() -> dict[str, str]:
    return {"status": "ok", "service": "vrp-solver"}


@app.post("/solve", response_model=SolverResponse)
async def solve(request: SolverRequest) -> SolverResponse:
    if not request.pois:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="pois must not be empty"
        )
    logger.info(
        "Solving itinerary: pois=%d days=%d strategy=%s",
        len(request.pois),
        request.constraints.travel_days,
        request.strategy,
    )
    return _solver.solve(request)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "vrp_solver_service.main:app",
        host="0.0.0.0",  # nosec B104
        port=8001,
        reload=False,
    )
