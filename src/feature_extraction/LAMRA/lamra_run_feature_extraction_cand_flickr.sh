#!/usr/bin/env bash
set -euo pipefail

genir_dir="/home/"
SRC="$genir_dir/src"


MBEIR_DATA_DIR="/data/Flickr30k/"

export PYTHONPATH="$SRC"
export CUDA_VISIBLE_DEVICES=0,1,2,3
NPROC=4

MODEL_NAME_OR_PATH="/data/checkpoint/LamRA-Ret/"
ORIGINAL_MODEL_ID="$MODEL_NAME_OR_PATH"
DTYPE="bf16"

EXTRACT_PY="$SRC/feature_extraction/LAMRA/lamra_feature_extraction_train.py"
CFG="$SRC/feature_extraction/LAMRA/lamra_extract_train.yaml"


SAVE_ROOT="/data/dig/embed/lamra/flickr"
QUERY_DIR="$SAVE_ROOT/test"
POOL_DIR="$SAVE_ROOT/cand"

mkdir -p "$QUERY_DIR" "$POOL_DIR"


DATASETS=(
  # flickr30k_task0 (text -> image)
  "flickr30k_task0_test|query/test/mbeir_flickr30k_task0_test.jsonl|cand_pool/mbeir_flickr30k_task0_test_cand_pool.jsonl"
)

MASTER_PORT_BASE=23350
i=0

for item in "${DATASETS[@]}"; do
  IFS="|" read -r TASK QUERY_PATH POOL_PATH <<< "$item"


  query_base=$(basename "$QUERY_PATH" .jsonl)
  pool_base=$(basename "$POOL_PATH" .jsonl)
  pool_task=${pool_base%_cand_pool}

  OUT_QUERY_NAME="${query_base}_dict.pt"
  OUT_POOL_NAME="${pool_task}_cand_pool_dict.pt"

  MASTER_PORT=$((MASTER_PORT_BASE + i))

  echo "============================================================"
  echo "[RUN] $TASK"
  echo "  cfg        = $CFG"
  echo "  mbeir_dir  = $MBEIR_DATA_DIR"
  echo "  query_path = $QUERY_PATH -> $QUERY_DIR/$OUT_QUERY_NAME"
  echo "  pool_path  = $POOL_PATH  -> $POOL_DIR/$OUT_POOL_NAME"
  echo "  model      = $MODEL_NAME_OR_PATH"
  echo "  dtype      = $DTYPE"
  echo "  gpus       = $CUDA_VISIBLE_DEVICES (nproc=$NPROC)"
  echo "  port       = $MASTER_PORT"
  echo "============================================================"

  python3 -m torch.distributed.run --nproc_per_node="$NPROC" --master_port="$MASTER_PORT" \
    "$EXTRACT_PY" \
    --config_path "$CFG" \
    --mbeir_data_dir "$MBEIR_DATA_DIR" \
    --save_dir "$SAVE_ROOT" \
    --query_save_dir "$QUERY_DIR" \
    --pool_save_dir "$POOL_DIR" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --original_model_id "$ORIGINAL_MODEL_ID" \
    --dtype "$DTYPE" \
    --do_query --do_pool \
    --query_data_path "$QUERY_PATH" \
    --cand_pool_path  "$POOL_PATH" \
    --query_out "$OUT_QUERY_NAME" \
    --pool_out  "$OUT_POOL_NAME"

  i=$((i + 1))
done

echo "============================================================"
echo "[DONE] Flickr30k embeddings"
echo "  query: $QUERY_DIR/"
echo "  cand:  $POOL_DIR/"
echo "============================================================"
