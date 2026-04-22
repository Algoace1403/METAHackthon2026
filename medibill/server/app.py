"""FastAPI server for MediBill-Env.

Uses OpenEnv's ``create_app`` to auto-generate all required endpoints:
``/health``, ``/metadata``, ``/schema``, ``/reset``, ``/step``, ``/state``,
``/ws``, ``/mcp``, ``/docs``.

Unlike Round 1 this module does not mount a Gradio dashboard — the hero
mechanic is inspected via API responses and the exploit-test suite, not
a live UI. That keeps the Docker image smaller and the HF Space cold-start
faster, both of which matter for a live demo.
"""

from __future__ import annotations

import os

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


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
