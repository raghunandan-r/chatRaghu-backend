"""Evaluation service for ChatRaghu"""

import sys as _sys  # noqa: E402

__version__ = "1.1.1"

# Early imports for shim targets
from . import config as _cfg  # noqa: E402
from . import models as _mdl  # noqa: E402
from . import utils as _utl  # noqa: E402

# Backward-compatibility shims
_sys.modules.setdefault("config", _cfg)
_sys.modules.setdefault("models", _mdl)
_sys.modules.setdefault("utils", _utl)

try:
    from .utils import logger as _lg  # noqa: E402

    _sys.modules.setdefault("utils.logger", _lg)
except Exception:
    pass

# Import evaluators after shims are ready
from . import evaluators as _eval_pkg  # noqa: E402

_sys.modules.setdefault("evaluators", _eval_pkg)

from . import storage as _stg  # noqa: E402

_sys.modules.setdefault("storage", _stg)

# Public API (these can stay without noqa since they're after setup)

try:
    from .evaluators import EVALUATOR_REGISTRY
except (ModuleNotFoundError, ImportError):
    EVALUATOR_REGISTRY = {}
