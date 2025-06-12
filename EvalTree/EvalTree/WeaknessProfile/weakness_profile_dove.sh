#!/bin/bash

export PYTHONPATH=$(dirname $(pwd))

# Set the results path for the model you're analyzing
RESULTS_PATH="real/Llama-3.1-8B-Instruct"

# Set the tree path (relative to Datasets/MMLU/EvalTree, no .bin/.json extension)
TREE_PATH="stage3-RecursiveClustering/[split=full]_[annotation=gpt-4o-mini]_[embedding=text-embedding-3-small]_[max-children=10]"

# Step 1: Compute confidence intervals
python3 -m EvalTree.WeaknessProfile.confidence_interval \
  --dataset MMLU \
  --tree_path "$TREE_PATH" \
  --results_path "$RESULTS_PATH"

# Step 2: Extract weakness profiles
python3 -m EvalTree.WeaknessProfile.extract_weaknesses \
  --dataset MMLU \
  --tree_path "$TREE_PATH" \
  --results_path "$RESULTS_PATH" \
  --description_model gpt-4o-mini \
  --direction lower \
  --alpha 0.05 \
  --threshold 0.4
