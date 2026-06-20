"""Download endpoints for generated itinerary artifacts (PDF / Excel)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from agents.output_format import EXCEL_DIR, PDF_DIR

router = APIRouter(prefix="/download", tags=["download"])


ALLOWED_DIRS: dict[Literal["pdfs", "excel"], str] = {
    "pdfs": PDF_DIR,
    "excel": EXCEL_DIR,
}


@router.get("/pdfs/{filename}")
async def download_pdf(filename: str) -> FileResponse:
    """Download a generated PDF itinerary."""
    return _serve_file("pdfs", filename, media_type="application/pdf")


@router.get("/excel/{filename}")
async def download_excel(filename: str) -> FileResponse:
    """Download a generated Excel itinerary."""
    return _serve_file(
        "excel",
        filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


def _serve_file(kind: Literal["pdfs", "excel"], filename: str, media_type: str) -> FileResponse:
    """Serve a file from the allowed output directory, preventing path traversal."""
    base_dir = ALLOWED_DIRS[kind]
    # Reject any filename containing path separators or parent references up front.
    if not filename or filename != os.path.basename(filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    if ".." in filename or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    file_path = (Path(base_dir) / filename).resolve()
    base_resolved = Path(base_dir).resolve()
    if not str(file_path).startswith(str(base_resolved)):
        raise HTTPException(status_code=400, detail="Invalid filename")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=filename,
    )
