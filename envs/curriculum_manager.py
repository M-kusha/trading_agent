"""
Compatibility shim.

Older code (and some tests/tools) import the curriculum manager from
`envs.curriculum_manager`. The implementation lives in the curriculum package;
prefer importing from `envs.curriculum.curriculum_manager`.
"""

from envs.curriculum.curriculum_manager import *  # noqa: F403

