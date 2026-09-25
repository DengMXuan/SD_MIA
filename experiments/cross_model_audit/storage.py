"""Cross-model audit uses the shared experiment lifecycle layout."""
from experiments.paths import AUDITS, prepare_audit_cache

OUTPUT_ROOT = AUDITS / 'cross_model_condition_seed_v1/tasks'
CACHE_ROOT = AUDITS / 'cross_model_condition_seed_v1/intermediate'
