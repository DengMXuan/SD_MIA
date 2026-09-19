"""Single source of truth for the controlled accept-only data roles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ControlledDataContract:
    """Record counts for training, detector fitting, calibration, and testing."""

    members: int = 2000
    nonmembers: int = 2000
    draft_auxiliary: int = 2000
    audit_auxiliary: int = 600
    detector_train: int = 320
    detector_validation: int = 80
    calibration: int = 200

    def __post_init__(self) -> None:
        values = (
            self.members,
            self.nonmembers,
            self.draft_auxiliary,
            self.audit_auxiliary,
            self.detector_train,
            self.detector_validation,
            self.calibration,
        )
        if any(value <= 0 for value in values):
            raise ValueError("controlled data counts must be positive")
        assigned = self.detector_train + self.detector_validation + self.calibration
        if assigned != self.audit_auxiliary:
            raise ValueError(
                "detector train, validation, and calibration counts must exhaust "
                "audit_auxiliary"
            )

    @property
    def shared_manifest_counts(self) -> dict[str, int]:
        """Names used by schema-v3 shared split manifests."""
        return {
            "member": self.members,
            "nonmember": self.nonmembers,
            "auxiliary": self.draft_auxiliary,
            "audit_auxiliary": self.audit_auxiliary,
        }

    @property
    def role_counts(self) -> dict[str, int]:
        return {
            "member": self.members,
            "nonmember": self.nonmembers,
            "draft_auxiliary": self.draft_auxiliary,
            "audit_auxiliary": self.audit_auxiliary,
        }

    @property
    def audit_partition_counts(self) -> dict[str, int]:
        return {
            "train": self.detector_train,
            "validation": self.detector_validation,
            "calibration": self.calibration,
        }

    @property
    def deployment_records(self) -> int:
        return self.audit_auxiliary + self.members + self.nonmembers


DEFAULT_DATA_CONTRACT = ControlledDataContract()
