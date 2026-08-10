"""The public API stays documented and stays re-exported.

Docstring coverage was cleaned up once at M5; without a test it would decay by
the next component added, and "docstrings on all public API" would quietly
become "docstrings on the parts written before July 2026". So the audit runs on
every commit instead of having been run once.

"Public" means: every module below, everything its ``__all__`` names, and every
public method, property and ``__call__`` on the classes it exports. Private
helpers are exempt — they are where the reasoning lives in comments instead.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil
from types import ModuleType
from typing import Any

import pytest

import spifil

SKIP_REASON = {
    "spifil.cli": (
        "a Hydra entry point whose public surface is the console script, "
        "not names other code imports"
    ),
    "spifil.conf": (
        "not code — the YAML config tree, a module only because Hydra "
        "resolves a packaged config_path as one"
    ),
}
"""Modules exempt from the ``__all__`` audit, and why. Kept as reasons rather
than a bare set so a skip in the report explains itself."""

SKIP = set(SKIP_REASON)


def all_modules() -> list[ModuleType]:
    """Every module in the package, imported."""
    found = [spifil]
    for info in pkgutil.walk_packages(spifil.__path__, prefix="spifil."):
        found.append(importlib.import_module(info.name))
    return found


def documented_members(module: ModuleType) -> list[tuple[str, Any]]:
    """``(name, object)`` for everything the module publishes, methods included."""
    members: list[tuple[str, Any]] = []
    for attribute in getattr(module, "__all__", []):
        obj = getattr(module, attribute)
        members.append((f"{module.__name__}.{attribute}", obj))
        if not inspect.isclass(obj):
            continue
        for name, member in vars(obj).items():
            if name.startswith("_") and name != "__call__":
                continue
            if isinstance(member, property):
                members.append((f"{module.__name__}.{attribute}.{name}", member.fget))
            elif inspect.isfunction(member):
                members.append((f"{module.__name__}.{attribute}.{name}", member))
    return members


@pytest.mark.parametrize("module", all_modules(), ids=lambda m: m.__name__)
def test_every_module_has_a_docstring(module: ModuleType) -> None:
    """Module docstrings carry the *why*; they are the first thing read."""
    assert inspect.getdoc(module), f"{module.__name__} has no module docstring"


@pytest.mark.parametrize("module", all_modules(), ids=lambda m: m.__name__)
def test_every_public_name_is_documented(module: ModuleType) -> None:
    undocumented = [
        name for name, obj in documented_members(module) if not inspect.getdoc(obj)
    ]

    assert not undocumented, f"undocumented public API: {', '.join(undocumented)}"


@pytest.mark.parametrize("module", all_modules(), ids=lambda m: m.__name__)
def test_every_module_declares_what_it_exports(module: ModuleType) -> None:
    """``__all__`` is what makes "public" a decision rather than an accident."""
    if module.__name__ in SKIP:
        pytest.skip(f"{module.__name__} is exempt: {SKIP_REASON[module.__name__]}")

    assert hasattr(module, "__all__"), f"{module.__name__} declares no __all__"


def test_the_package_reexports_the_names_it_advertises() -> None:
    """Everything in ``spifil.__all__`` actually resolves.

    A stale entry is an ``ImportError`` for whoever follows the README, and
    nothing else in the suite imports the package by ``__all__``.
    """
    missing = [name for name in spifil.__all__ if not hasattr(spifil, name)]

    assert not missing, f"spifil.__all__ names things that do not exist: {missing}"


def test_the_declared_version_matches_the_package_metadata() -> None:
    """``spifil.__version__`` and ``pyproject.toml`` must agree.

    They are declared separately and drift silently: a release bumps one, and
    the other keeps reporting the old number in every exported bundle and bug
    report until somebody notices.
    """
    from importlib.metadata import version

    assert spifil.__version__ == version("spifil")


def test_the_package_all_has_no_duplicates() -> None:
    """A duplicate is a merge artifact, and nothing else would flag it.

    Ordering is not asserted: the list groups acronyms first and
    then sorts, which is ruff's ``RUF022`` convention rather than
    :func:`sorted`'s, and ``RUF`` is not among the enabled rule sets.
    """
    duplicated = {name for name in spifil.__all__ if spifil.__all__.count(name) > 1}

    assert not duplicated, f"spifil.__all__ repeats: {sorted(duplicated)}"
