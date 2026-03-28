"""FastAPI server for DataClean-Env.

Uses OpenEnv's create_app() to auto-generate all required endpoints:
/health, /metadata, /schema, /reset, /step, /state, /ws, /mcp, /docs
"""

from openenv.core.env_server.http_server import create_app

from dataclean_env.models import DataCleanAction, DataCleanObservation
from dataclean_env.server.environment import DataCleanEnvironment

app = create_app(
    DataCleanEnvironment,
    DataCleanAction,
    DataCleanObservation,
    env_name="dataclean_env",
    max_concurrent_envs=1,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
