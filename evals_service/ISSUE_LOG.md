# CRITICAL ISSUE LOG: Import System Failure & Recovery Plan

**Date**: 2025-01-08
**Status**: ACTIVE - Service Currently Broken
**Impact**: Docker container fails to start, integration tests broken

## Original Working State (Before Issues)

The service had a sophisticated **shim-based import system** in `__init__.py` that allowed modules to use simple imports:

```python
# Files could use:
from config import config
from models import EnrichedNodeExecutionLog
from utils.logger import logger

# Instead of verbose:
from evals_service.config import config
from evals_service.models import EnrichedNodeExecutionLog
from evals_service.utils.logger import logger
```

**How it worked**: `sys.modules.setdefault()` calls in `__init__.py` mapped short names to full module paths.

**Status before issues**: ✅ Unit tests passing, ✅ Integration tests passing, ✅ Docker working, ✅ Railway deployment working

## The Cascade of Failures

### Issue 1: Ruff E402 Linting Errors
**Date**: Started yesterday
**Problem**: Ruff complained about "Module level import not at top of file" (E402) because shim setup required runtime code before imports.

**Example error**:evals_service/init.py:24:1: E402 Module level import not at top of file


### Failed Fix 1: Import Reordering
**What was tried**: Moving imports around in `__init__.py` to satisfy Ruff
**Result**: Broke the carefully orchestrated shim setup order
**New problems**: `ModuleNotFoundError` in tests

### Failed Fix 2: Partial Absolute Import Conversion
**What was tried**: Changed some files from `from config import config` to `from evals_service.config import config`
**Result**: Created inconsistency - some files worked, others didn't
**New problems**: More `ModuleNotFoundError` in different files

### Failed Fix 3: Complete Shim Removal
**What was tried**: Removed all `sys.modules` logic, forced absolute imports everywhere
**Result**: Local tests worked, Docker container failed
**New problems**: `ModuleNotFoundError: No module named 'evals_service'` in Docker

### Failed Fix 4: Docker Symlink Hack
**What was tried**: Added `ln -s . evals_service` in Dockerfile
**Result**: Symlinks don't work reliably with Python's import system
**New problems**: Service still couldn't start

### Failed Fix 5: Docker Structure Overhaul
**What was tried**: Changed Docker COPY structure, modified CMD to use `evals_service.app:app`
**Result**: Broke Railway deployment compatibility
**New problems**: `Error: Invalid value for '--port': '${API_PORT:-8001}' is not a valid integer`

### Failed Fix 6: CMD Format Fix
**What was tried**: Changed CMD to use shell form for variable expansion
**Result**: Still fails due to wrong module structure
**Current status**: Service container cannot start

## Root Cause Analysis

**The real issue**: We treated **Ruff E402 warnings as errors** when they were just **cosmetic linting issues** in a **working system**.

**What should have been done**: Use `# noqa: E402` comments or configure Ruff to ignore E402 in `__init__.py` files.

**What was done instead**: Dismantled a working, sophisticated import system to satisfy a linter.

## Recovery Plan (Approved Approach)

### Objective
Restore the original working shim system while satisfying Ruff with minimal `# noqa: E402` suppressions.

### Step 1: Restore Shim System in `__init__.py`
```python
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
from .models import EnrichedNodeExecutionLog, ConversationFlow
from .evaluators.models import (
    RelevanceCheckEval,
    GenerateWithPersonaEval,
    NodeEvaluation,
)
from .storage import (
    create_storage_backend,
    LocalStorageBackend,
    GCSStorageBackend,
)
from .config import get_config, Config

try:
    from .evaluators import EVALUATOR_REGISTRY
except (ModuleNotFoundError, ImportError):
    EVALUATOR_REGISTRY = {}

__all__ = [
    "EnrichedNodeExecutionLog",
    "ConversationFlow",
    "RelevanceCheckEval",
    "GenerateWithPersonaEval",
    "NodeEvaluation",
    "EVALUATOR_REGISTRY",
    "create_storage_backend",
    "LocalStorageBackend",
    "GCSStorageBackend",
    "get_config",
    "Config",
]
```

### Step 2: Revert All Files to Shim-Based Imports
**Files requiring reversion**:
- `evaluators/relevance_check.py`
- `evaluators/generate_with_persona.py`
- `evaluators/generate_with_context.py`
- `evaluators/__init__.py`
- `queue_manager.py`
- `run_evals.py`
- `app.py`
- `storage.py`
- `tests/conftest.py`
- `tests/test_unit.py`

**Change pattern**: Revert FROM `from evals_service.X import Y` back TO `from X import Y`

### Step 3: Restore Original Docker Configuration
```dockerfile
WORKDIR /app

# Copy everything to app root (revert to original)
COPY . ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir -e .

# Create storage directories
RUN useradd -ms /bin/bash appuser && \
    mkdir -p /app/audit_data /app/eval_results && \
    chown -R appuser:appuser /app/audit_data /app/eval_results && \
    chmod -R 755 /app/audit_data /app/eval_results

USER appuser

# Copy entrypoint
COPY --chown=appuser:appuser entrypoint.sh .
RUN chmod +x entrypoint.sh

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

ENTRYPOINT ["./entrypoint.sh"]

# Restore original CMD with Railway compatibility
CMD ["sh", "-c", "uvicorn app:app --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8001}"]
```

### Step 4: Fix Test Module Reloads
Revert test reload logic back to shim names:
```python
if "config" in sys.modules:
    importlib.reload(sys.modules["config"])
if "storage" in sys.modules:
    importlib.reload(sys.modules["storage"])
```

## Impact Assessment Matrix

| Component | Original State | Current State | After Recovery |
|-----------|---------------|---------------|----------------|
| Ruff Linting | ❌ E402 errors | ✅ Passes | ✅ Passes (with noqa) |
| Unit Tests | ✅ Passing | ❌ Import errors | ✅ Should pass |
| Integration Tests | ✅ Passing | ❌ Service down | ✅ Should pass |
| Docker Container | ✅ Starts | ❌ Import errors | ✅ Should start |
| Railway Deploy | ✅ Working | ❌ Broken | ✅ Should work |

## Current State Analysis
```bash
# Current Docker error:
# ModuleNotFoundError: No module named 'evals_service'
# File "/app/app.py", line 11, in <module>
# from evals_service.models import (
```

## Key Lessons
1. **Don't break working systems for linting cosmetics**
2. **Use linter configuration/suppressions instead of architectural changes**
3. **Test changes in all environments (local, Docker, CI)**
4. **Understand the system before modifying it**
5. **Rollback strategy is critical for complex changes**

## Context Preservation Notes
- User cannot rollback git history (would lose evaluator refactor)
- Must restore functionality through targeted fixes
- Railway deployment needs specific CMD format with variable expansion
- Original shim system was sophisticated and working
- Current Docker logs show: `ModuleNotFoundError: No module named 'evals_service'`

## Execution Plan
1. **Save this file** as `evals_service/ISSUE_LOG.md`
2. **Step 1**: Restore `__init__.py` shim system
3. **Step 2**: Revert import statements in all listed files
4. **Step 3**: Restore Dockerfile structure
5. **Step 4**: Fix test module reloads
6. **Test each step** before proceeding to next

**This document preserves context for any future debugging sessions.**

---

## Agent's Recovery Log & Analysis (2025-01-09)

This section documents the agent's actions and analysis while executing the recovery plan.

### Issues Identified in Pending Files

1.  **`Dockerfile`**: The current version uses a multi-directory structure (`COPY . ./evals_service/`) which, combined with the other import issues, caused `ModuleNotFoundError`. The recovery plan correctly identifies that this needs to be reverted to a flat structure (`COPY . ./`) with `PYTHONPATH` set to `/app`. This allows the shimmed import system to work correctly inside the container.

2.  **`tests/conftest.py`**: This file is responsible for setting up the test environment.
    -   **Problem**: Its `_setup_imports` function tries to add an incorrect path (`.../evals-service`) to `sys.path` and uses absolute imports (`from evals_service.models...`). This breaks when the shim system is active.
    -   **Solution**: The file must be modified to correctly add the project root to `sys.path`, import `evals_service` to activate the `__init__.py` shims, and then use direct imports like `from models import ...`.

3.  **`tests/test_unit.py`**: This file contains several problems.
    -   **Absolute Imports**: It uses absolute imports (`from evals_service.evaluators.base...`) instead of the required shim-based imports.
    -   **Incorrect Reloads**: The `importlib.reload` calls reference the full module path (`sys.modules["evals_service.config"]`) instead of the shim name (`sys.modules["config"]`), which is explicitly what Step 4 of the plan is meant to fix.
    -   **Agent Overreach**: My previous analysis suggested removing the `EVALS_SERVICE_AVAILABLE` flag and its associated `skipif` decorators. This was an error on my part. It is an unnecessary change that goes beyond the scope of the immediate recovery plan. The safeguard should remain in place until the system is fully validated.

### Comparison of Initial Plan vs. Actions

-   **Plan**: Restore shim system, revert imports, restore Dockerfile, fix test reloads.
-   **Actions Taken**:
    -   **Step 1 (`__init__.py`)**: Completed successfully.
    -   **Step 2 (Service Files)**: Completed successfully for `app`, `storage`, `queue_manager`, `run_evals`, and `evaluators/*.py`.
    -   **Step 2 & 4 (Test Files)**: **Previously Failed.** My attempts were too broad and flawed. The new approach will be surgical and safe, adhering strictly to the plan.
    -   **Step 3 (`Dockerfile`)**: **Pending user approval.** The proposed change is correct per the plan.

My next steps will be to propose the correct, minimal changes for the remaining files, starting with the test files that I previously mishandled. I will not remove any safeguards.
