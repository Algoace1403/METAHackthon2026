"""DataClean-Env test package.

Installs openenv mock before any other dataclean_env imports can trigger.
"""

from __future__ import annotations

import sys
from types import ModuleType


def _install_openenv_mock() -> None:
    if "openenv" in sys.modules:
        return

    class _Action:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Observation:
        done: bool = False
        reward = None
        metadata: dict = {}
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _State:
        episode_id: str = ""
        step_count: int = 0
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _Environment:
        def __init__(self): pass
        def __class_getitem__(cls, item): return cls

    class _EnvironmentMetadata:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    class _EnvClient:
        def __init__(self, *args, **kwargs): pass
        def __class_getitem__(cls, item): return cls

    class _StepResult:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    openenv = ModuleType("openenv")
    openenv_core = ModuleType("openenv.core")
    openenv_core_env_server = ModuleType("openenv.core.env_server")
    openenv_core_env_server_types = ModuleType("openenv.core.env_server.types")
    openenv_core_env_client = ModuleType("openenv.core.env_client")
    openenv_core_client_types = ModuleType("openenv.core.client_types")

    openenv_core_env_server.Action = _Action
    openenv_core_env_server.Observation = _Observation
    openenv_core_env_server.State = _State
    openenv_core_env_server.Environment = _Environment
    openenv_core_env_server_types.EnvironmentMetadata = _EnvironmentMetadata
    openenv_core_env_client.EnvClient = _EnvClient
    openenv_core_client_types.StepResult = _StepResult

    openenv.core = openenv_core
    openenv_core.env_server = openenv_core_env_server
    openenv_core.env_client = openenv_core_env_client
    openenv_core.client_types = openenv_core_client_types

    sys.modules["openenv"] = openenv
    sys.modules["openenv.core"] = openenv_core
    sys.modules["openenv.core.env_server"] = openenv_core_env_server
    sys.modules["openenv.core.env_server.types"] = openenv_core_env_server_types
    sys.modules["openenv.core.env_client"] = openenv_core_env_client
    sys.modules["openenv.core.client_types"] = openenv_core_client_types


_install_openenv_mock()
