"""Pytest plugin that mocks openenv-core. Load with: pytest -p tests._mock_openenv"""

from __future__ import annotations

import sys
from types import ModuleType


def _install() -> None:
    if "openenv" in sys.modules:
        return

    class _Action:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _Observation:
        done: bool = False
        reward = None
        metadata: dict = {}

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _State:
        episode_id: str = ""
        step_count: int = 0

        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _Environment:
        def __init__(self):
            pass

        def __class_getitem__(cls, item):
            return cls

    class _EnvironmentMetadata:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    class _EnvClient:
        def __init__(self, *a, **kw):
            pass

        def __class_getitem__(cls, item):
            return cls

    class _StepResult:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    names = [
        "openenv",
        "openenv.core",
        "openenv.core.env_server",
        "openenv.core.env_server.types",
        "openenv.core.env_client",
        "openenv.core.client_types",
    ]
    mods = {n: ModuleType(n) for n in names}
    for n, m in mods.items():
        sys.modules[n] = m

    mods["openenv"].core = mods["openenv.core"]
    mods["openenv.core"].env_server = mods["openenv.core.env_server"]
    mods["openenv.core"].env_client = mods["openenv.core.env_client"]
    mods["openenv.core"].client_types = mods["openenv.core.client_types"]

    mods["openenv.core.env_server"].Action = _Action
    mods["openenv.core.env_server"].Observation = _Observation
    mods["openenv.core.env_server"].State = _State
    mods["openenv.core.env_server"].Environment = _Environment
    mods["openenv.core.env_server.types"].EnvironmentMetadata = _EnvironmentMetadata
    mods["openenv.core.env_client"].EnvClient = _EnvClient
    mods["openenv.core.client_types"].StepResult = _StepResult


_install()
