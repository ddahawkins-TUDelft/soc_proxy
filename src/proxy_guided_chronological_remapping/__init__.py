"""Proxy-guided chronological remapping.

The package provides lightweight post-TSA remapping methods that use a fixed
ex-ante SoC-proxy signal to improve chronological reconstruction without
solving a global assignment optimisation problem.
"""

from .greedy import (
    ProxyRemapDiagnostics,
    ProxyRemapResult,
    greedy_proxy_chronology_remap,
    proxy_error_metrics,
    reconstruct_proxy_delta,
)
from .tsam_adapter import (
    TsamProxyChronologyInputs,
    prepare_tsam_proxy_chronology_inputs,
)

__all__ = [
    "ProxyRemapDiagnostics",
    "ProxyRemapResult",
    "TsamProxyChronologyInputs",
    "greedy_proxy_chronology_remap",
    "prepare_tsam_proxy_chronology_inputs",
    "proxy_error_metrics",
    "reconstruct_proxy_delta",
]
