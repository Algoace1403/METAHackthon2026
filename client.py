"""Root-level re-export shim — see ``__init__.py`` docstring.

The actual client lives at ``medibill/client.py``. Import from there.
"""

from medibill.client import MediBillEnv  # noqa: F401
