"""ExPLoRe-VTLA: reality-coupled loss routing for embodied multimodal learning."""

from .contact_dynamics import ContactParameters, ContactWorld
from .contracts import AuthorityLevel, SignalHealth, VTLAConfig, VTLASequence
from .diagnostics import RoutingDiagnostics
from .model import ExPLoReVTLA
from .router import LossCoupledRouter, loss_coupled_reduce

__all__ = [
    "ContactParameters",
    "ContactWorld",
    "AuthorityLevel",
    "SignalHealth",
    "VTLAConfig",
    "VTLASequence",
    "ExPLoReVTLA",
    "LossCoupledRouter",
    "RoutingDiagnostics",
    "loss_coupled_reduce",
]

__version__ = "1.0.0"
