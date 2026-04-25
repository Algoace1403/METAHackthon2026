"""Root-level re-export shim for OpenEnv's push validator.

The OpenEnv CLI's ``validate_env_structure`` requires ``__init__.py``,
``client.py``, and ``models.py`` at the env root. The actual package code
lives under ``medibill/`` — these three files just re-export from there.

Do NOT import from these shims in production code; import from ``medibill.*``
directly. The shims exist only to satisfy the OpenEnv CLI validator and
will be removed when that validator is loosened.
"""

from medibill import (  # noqa: F401
    AGENT_ACTION_TYPES,
    ClaimPreview,
    DriftRecord,
    MediBillAction,
    MediBillEnv,
    MediBillObservation,
    MediBillState,
    ToolResult,
    __version__,
)
