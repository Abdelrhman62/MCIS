#!/bin/bash

# Create directories
mkdir -p results/diagnostics
mkdir -p results/final_evals

# Initialize summary CSV
SUMMARY_FILE="results/final_evals/final_seed_fold_summary.csv"
echo "seed,fold,best_epoch,early_stop_metric" > $SUMMARY_FILE

for seed in 42 123 456; do
  for fold in 0 1 2 3 4; do
    echo "=================================================="
    echo "Starting Seed $seed | Fold $fold"
    echo "=================================================="
    
    EXP_NAME="MCIS_Best_seed${seed}_fold${fold}"
    LOG_FILE="results/diagnostics/mcis_best_seed${seed}_fold${fold}.log"
    EVAL_FILE="results/final_evals/eval_seed${seed}_fold${fold}.md"
    
    # 1. Train the model
    PYTHONPATH=/root/MCIS python -m src.train \
      --config configs/MCIS_Best.yaml \
      --fold $fold \
      --device cuda \
      --experiment-name $EXP_NAME \
      --seed $seed \
      2>&1 | tee $LOG_FILE
      
    # Extract Best Epoch and Metric for quick monitoring
    BEST_EPOCH=$(grep "Done." $LOG_FILE | sed -n 's/.*best_epoch=\([0-9]*\).*/\1/p')
    BEST_METRIC=$(grep "Done." $LOG_FILE | sed -n 's/.*best_metric=\([0-9.]*\).*/\1/p')
    
    echo "$seed,$fold,$BEST_EPOCH,$BEST_METRIC" >> $SUMMARY_FILE
    
    # 2. Run the comprehensive evaluation we just built
    # The checkpoint is saved dynamically using the experiment name
    CKPT_PATH="checkpoints/${EXP_NAME}/best.pt"
    
    if [ -f "$CKPT_PATH" ]; then
      PYTHONPATH=/root/MCIS python scripts/run_full_evaluation.py \
        --checkpoint $CKPT_PATH \
        --config configs/MCIS_Best.yaml \
        --split val_fold \
        --fold $fold \
        --device cuda \
        --output $EVAL_FILE
      echo "Evaluation saved to $EVAL_FILE"
    else
      echo "ERROR: Checkpoint not found at $CKPT_PATH"
    fi
    
  done
done

echo "=================================================="
echo "All 15 runs complete! Summary saved to $SUMMARY_FILE"
