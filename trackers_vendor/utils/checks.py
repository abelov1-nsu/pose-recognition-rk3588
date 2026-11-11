"""No-op check_requirements so vendored code doesn't try to pip-install at runtime."""
from __future__ import annotations


def check_requirements(requirements=None, exclude=(), install=True, cmds="", action=None):
    """Vendored no-op. The env is fixed at install time; nothing to verify."""
    return True


__all__ = ["check_requirements"]