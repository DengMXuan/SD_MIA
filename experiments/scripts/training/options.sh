# Shared training command parser. GPUS is initially populated from MATRIX_GPUS.
MODE=dry-run
CONDITION_ARGS=()
TRAIN_WORKERS=''
while (( $# )); do
  case "$1" in
    run) MODE=run; shift ;;
    dry-run|--dry-run) MODE=dry-run; shift ;;
    status|--status) MODE=status; shift ;;
    preflight|--preflight-only) MODE=preflight; shift ;;
    --gpu)
      (( $# >= 2 )) || { echo '--gpu needs an index' >&2; exit 2; }
      GPUS=("$2"); shift 2 ;;
    --gpus)
      GPUS=(); shift
      while (( $# )) && [[ "$1" != -* ]]; do GPUS+=("$1"); shift; done ;;
    --workers)
      (( $# >= 2 )) || { echo '--workers needs a count' >&2; exit 2; }
      TRAIN_WORKERS=$2; shift 2 ;;
    --_condition) MODE=condition; CONDITION_ARGS=("${@:2}"); GPUS=(0); break ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
done
if (( ${#GPUS[@]} == 0 )); then echo 'select at least one GPU' >&2; exit 2; fi
declare -A TRAIN_SEEN_GPUS=()
for gpu in "${GPUS[@]}"; do
  if [[ ! "$gpu" =~ ^(0|[1-9][0-9]*)$ ]] || [[ -v "TRAIN_SEEN_GPUS[$gpu]" ]]; then
    echo 'choose unique nonnegative GPU indices' >&2; exit 2
  fi
  TRAIN_SEEN_GPUS[$gpu]=1
done
if [[ -n "$TRAIN_WORKERS" ]]; then
  if [[ ! "$TRAIN_WORKERS" =~ ^[1-9][0-9]*$ ]] || (( TRAIN_WORKERS > ${#GPUS[@]} )); then
    echo '--workers must be between 1 and the selected GPU count' >&2; exit 2
  fi
  GPUS=("${GPUS[@]:0:TRAIN_WORKERS}")
fi
