"""FastAPI server for DataClean-Env.

Uses OpenEnv's create_app() to auto-generate all required endpoints:
/health, /metadata, /schema, /reset, /step, /state, /ws, /mcp, /docs

Mounts Gradio dashboard at root for interactive testing.
"""

import os

import gradio as gr
from openenv.core.env_server.http_server import create_app

from dataclean_env.models import DataCleanAction, DataCleanObservation
from dataclean_env.server.environment import DataCleanEnvironment
from dataclean_env.server.web_ui import build_ui

app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="dataclean_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "1")),
)

# Mount Gradio dashboard at root
gradio_app = build_ui()
app = gr.mount_gradio_app(app, gradio_app, path="/")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
