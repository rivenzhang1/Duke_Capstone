"""Variant registry for encoder architectures.

Design: docs/design.md §4.3; convention in CLAUDE.md, "Dispatch / registry pattern".

Each variant module exposes a fixed contract:

    NAME: str
    VERSION: int          # bump whenever behavior or output changes
    def build(config) -> ModelSpec

Variants are imported **lazily** (registering an architecture must not import torch until
that architecture is actually used) and duplicate registration **fails loud**. Results record
the variant name and VERSION so old runs stay interpretable.

Public surface:
    register(name, module_path) -> None
    get(name) -> ModelSpec builder (the variant module's `build` function)
    available() -> list[str]
"""

from __future__ import annotations

import importlib

_REGISTRY: dict[str, str] = {}


def register(name: str, module_path: str) -> None:
    if name in _REGISTRY and _REGISTRY[name] != module_path:
        raise ValueError(
            f"variant {name!r} already registered to {_REGISTRY[name]!r}, "
            f"refusing to overwrite with {module_path!r}"
        )
    _REGISTRY[name] = module_path


def get_module(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"unknown variant {name!r}; available: {available()}")
    module = importlib.import_module(_REGISTRY[name])
    if module.NAME != name:
        raise ValueError(f"{_REGISTRY[name]}.NAME == {module.NAME!r}, expected {name!r}")
    return module


def get(name: str):
    return get_module(name).build


def available() -> list[str]:
    return sorted(_REGISTRY)


# Variants register themselves here by name -> module path. Importing this module does not
# import torch; `get()` only imports the specific variant's module, and only when asked for.
register("dynamic_factor", "deeppace_net.models.dynamic_factor")
register("two_stage", "deeppace_net.models.two_stage")
