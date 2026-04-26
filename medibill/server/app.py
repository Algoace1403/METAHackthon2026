"""FastAPI server for MediBill-Env.

Uses OpenEnv's ``create_app`` to auto-generate all required endpoints:
``/health``, ``/metadata``, ``/schema``, ``/reset``, ``/step``, ``/state``,
``/ws``, ``/mcp``, ``/docs``. Adds two zero-cost extras for storytelling:

* ``GET /demo`` — static HTML viewer that replays a pre-recorded
  ``ScriptedDriftAwarePolicy`` rollout on ``hard_drift`` seed 42 with
  drift events visually highlighted. Lets a judge see the hero mechanic
  in action without invoking the env or pulling the trained adapter.
* ``GET /demo_trajectory.json`` — the trajectory file the viewer reads.

These are static files; they add zero per-request load on the env
worker and do not change the API surface. Cold-start cost is still
dominated by FastAPI startup.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi.responses import FileResponse
from openenv.core.env_server.http_server import create_app

from medibill.models import MediBillAction, MediBillObservation
from medibill.server.environment import MediBillEnvironment


app = create_app(
    MediBillEnvironment,
    MediBillAction,
    MediBillObservation,
    env_name="medibill",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "1")),
)


_STATIC = Path(__file__).resolve().parent / "static"


@app.get("/demo")
def demo_page() -> FileResponse:
    """Serve the static demo viewer."""
    return FileResponse(_STATIC / "demo.html", media_type="text/html")


@app.get("/demo_trajectory.json")
def demo_trajectory() -> FileResponse:
    """Serve the pre-recorded ScriptedDriftAwarePolicy trajectory."""
    return FileResponse(_STATIC / "demo_trajectory.json", media_type="application/json")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
