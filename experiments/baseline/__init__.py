"""Target-only membership-inference baselines for the controlled SFT runs."""

METHODS = (
    "loss",
    "min_k_prob",
    "min_k_pp",
    "recall",
    "icp_mia",
    "petal",
    "sead",
    "ws",
    "rs",
    "bt",
    "samia",
)

__all__ = ["METHODS"]
