#!/bin/bash

# Define the arrays of models and datasets
declare -a models=("CNN" "DANN" "ACDANN" "CORAL" "CDAN")
declare -a CWRU_sources=("CWRU_0" "CWRU_1" "CWRU_2" "CWRU_3")
declare -a jnu_sources=("JNU_0" "JNU_1" "JNU_2")

# Loop over each model for CWRU dataset

for model in "${models[@]}"; do
  for source in "${CWRU_sources[@]}"; do
    for target in "${CWRU_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target"
      fi
    done
  done
done

# Loop over each model for JNU dataset

for model in "${models[@]}"; do
  for source in "${jnu_sources[@]}"; do
    for target in "${jnu_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target"
      fi
    done
  done
done