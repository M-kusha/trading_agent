"""
Compatibility shim.

Older code (and some tests/tools) import curriculum helpers from
`envs.curriculum_config`. The implementation was moved to the curriculum
package; prefer importing from `envs.curriculum.curriculum_config`.
"""

from envs.curriculum.curriculum_config import *  # noqa: F403

