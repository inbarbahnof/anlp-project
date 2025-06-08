#!/bin/bash

set -e  # Exit on error
set -x  # Print each command

# Stage 1: Annotate
python -m EvalTree.stage1-CapabilityAnnotation.annotate --dataset MMLU

# Stage 2: Embedding
python -m EvalTree.stage2-CapabilityEmbedding.embedding --dataset MMLU

# Stage 3: Recursive Clustering
python -m EvalTree.stage3-RecursiveClustering.build \
  --dataset MMLU \
  --split full

# Stage 4: Capability Description
python -m EvalTree.stage4-CapabilityDescription.describe \
  --dataset MMLU \
  --tree_path stage3-RecursiveClustering/[split=full][annotation=gpt-4o-mini][embedding=text-embedding-3-small]_[max-children=10]