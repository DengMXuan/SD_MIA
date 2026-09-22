"""Resource units and one-dimensional experiment settings."""
from dataclasses import asdict, dataclass

QUERY_MULTIPLICITIES = (1, 2, 4, 8, 16)


def check_multiplicity(value: int) -> None:
    if type(value) is not int or value not in QUERY_MULTIPLICITIES:
        raise ValueError("query multiplicity must be one of 1, 2, 4, 8, 16")


@dataclass(frozen=True)
class AuxiliaryBudget:
    """Fitting includes training and validation, never language-model tuning."""

    fitting: int = 400
    calibration: int = 200
    validation: int | None = None

    def __post_init__(self):
        if type(self.fitting) is not int or type(self.calibration) is not int:
            raise ValueError("auxiliary sizes must be integers")
        validation = self.fitting // 5 if self.validation is None else self.validation
        if (type(validation) is not int or not 0 < validation < self.fitting
                or self.calibration < 1 or not 200 <= self.total <= 1600):
            raise ValueError("positive train/validation/calibration and total 200..1600 required")
        object.__setattr__(self, "validation", validation)

    @property
    def train(self):
        return self.fitting - self.validation

    @property
    def total(self):
        return self.fitting + self.calibration

    def to_dict(self):
        return {**asdict(self), "train": self.train, "total": self.total}


def calibration_curve():
    return tuple(AuxiliaryBudget(400, size) for size in (200, 600, 1200))


def fitting_curve():
    return tuple(AuxiliaryBudget(size, 200) for size in (400, 800, 1200))
