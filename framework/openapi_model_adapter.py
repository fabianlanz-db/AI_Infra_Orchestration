"""
Deprecated: import ``OpenApiModelClient``/``OpenApiModelConfig`` from
``framework.external_model_hooks`` instead. This module is kept as a
compatibility wrapper and will be removed in a future release.
"""

import warnings

from framework.external_model_hooks import OpenApiModelClient, OpenApiModelConfig

warnings.warn(
    "framework.openapi_model_adapter is deprecated; import from "
    "framework.external_model_hooks instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["OpenApiModelClient", "OpenApiModelConfig"]
