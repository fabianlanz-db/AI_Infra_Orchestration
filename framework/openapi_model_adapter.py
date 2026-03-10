"""
Compatibility wrapper for OpenAPI model adapters.

Canonical implementation now lives in `framework/external_model_hooks.py` to
keep external-model orchestration and adapter logic consolidated in one place.
"""

from framework.external_model_hooks import OpenApiModelClient, OpenApiModelConfig

__all__ = ["OpenApiModelClient", "OpenApiModelConfig"]
