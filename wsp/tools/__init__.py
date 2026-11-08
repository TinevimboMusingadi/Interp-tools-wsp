"""Tool sets exposed to the agent, one per experimental condition.

- ``c0``: paper baseline (python + generate + edit_report + submit)
- ``c1``: c0 + interpretability tools (capture, logit lens, attention, contrast directions,
  steering/ablation previews, neuron/direction readouts)
- ``c1_sae``: c1 + pretrained SAE feature tools (needs ``saes`` in the env config)
- ``c2``: c1_sae + tool-building (probes, SAE training, persistent helper library)
"""

from __future__ import annotations

from collections.abc import Callable


def _c0():
    from wsp.tools.base import c0_tools

    return c0_tools()


def _c1():
    from wsp.tools.interp import c1_tools

    return c1_tools()


def _c1_sae():
    from wsp.tools.interp import c1_tools_with_sae

    return c1_tools_with_sae()


def _c2():
    from wsp.tools.build import c2_tools

    return c2_tools()


TOOLSETS: dict[str, Callable] = {"c0": _c0, "c1": _c1, "c1_sae": _c1_sae, "c2": _c2}

__all__ = ["TOOLSETS"]
