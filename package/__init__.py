"""Evidence-backed catalog packaging for the Kimi-Linear serving engine."""

from .card import render_model_card
from .claims import (
    Claim,
    ClaimBasis,
    ClaimKind,
    Comparison,
    EvidenceArtifact,
    EvidenceReference,
    HardwareContext,
    RuntimeIdentity,
    render_claim,
)
from .definition import (
    BaseModelReference,
    DisclosureSet,
    HardwareRequirement,
    LicensePosition,
    LicenseStatus,
    ModificationDisclosure,
    PackageClass,
    PackageDefinition,
    build_kimi_linear_definition,
)

__all__ = [
    "BaseModelReference",
    "Claim",
    "ClaimBasis",
    "ClaimKind",
    "Comparison",
    "DisclosureSet",
    "EvidenceArtifact",
    "EvidenceReference",
    "HardwareContext",
    "HardwareRequirement",
    "LicensePosition",
    "LicenseStatus",
    "ModificationDisclosure",
    "PackageClass",
    "PackageDefinition",
    "RuntimeIdentity",
    "build_kimi_linear_definition",
    "render_claim",
    "render_model_card",
]
