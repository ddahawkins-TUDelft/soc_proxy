"""SoC Proxy public API."""

from ._margin import MarginDiagnostics
from .soc_proxy import SocProxyResult, generate_soc_proxy

__all__ = [
    "MarginDiagnostics",
    "SocProxyResult",
    "generate_soc_proxy",
]
