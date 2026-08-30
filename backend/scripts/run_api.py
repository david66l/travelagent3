"""Start the API with an event loop compatible with async Psycopg on Windows."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import uvicorn

# The wrapper is executed directly in local/CI environments where the project
# is not installed as a wheel. Keep its import behavior aligned with the Docker
# image's PYTHONPATH=/app/src contract.
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def main() -> None:
    uvicorn.run(
        "api.main:app",
        host=os.getenv("APP_HOST", "0.0.0.0"),
        port=int(os.getenv("APP_PORT", "8000")),
        loop="core.event_loop:psycopg_compatible_loop",
    )


if __name__ == "__main__":
    main()
