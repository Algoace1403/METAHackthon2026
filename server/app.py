"""FastAPI server for DataClean-Env.

Uses OpenEnv's create_app() to auto-generate all required endpoints:
/health, /metadata, /schema, /reset, /step, /state, /ws, /mcp, /docs
"""

import os

from fastapi.responses import RedirectResponse
from openenv.core.env_server.http_server import create_app

from dataclean_env.models import DataCleanAction, DataCleanObservation
from dataclean_env.server.environment import DataCleanEnvironment

app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="dataclean_env",
    max_concurrent_envs=int(os.environ.get("MAX_CONCURRENT_ENVS", "1")),
)


@app.get("/", include_in_schema=False)
def root():
    """Redirect root to API docs."""
    return RedirectResponse(url="/docs")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
