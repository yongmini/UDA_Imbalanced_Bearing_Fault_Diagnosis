#!/bin/bash

## cross-domain adaptation experiments


declare -a models=("CNN" )
declare -a CWRU_sources=("CWRU_0" "CWRU_1" "CWRU_2")
declare -a jnu_sources=("JNU_0" "JNU_1" "JNU_2")

# Loop over each model for CWRU dataset

for model in "${models[@]}"; do
  for source in "${CWRU_sources[@]}"; do
    for target in "${CWRU_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target" --Domain="CWRU"
      fi
    done
  done
done

# Loop over each model for JNU dataset

for model in "${models[@]}"; do
  for source in "${jnu_sources[@]}"; do
    for target in "${jnu_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target" --Domain="JNU"
      fi
    done
  done
done


## cross-domain adaptation experiments imabalanced data  normal 100% others 1%


# Loop over each model for CWRU dataset

for model in "${models[@]}"; do
  for source in "${CWRU_sources[@]}"; do
    for target in "${CWRU_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target" --Domain="CWRU_imba" --imba True
      fi
    done
  done
done



# Loop over each model for JNU dataset

for model in "${models[@]}"; do
  for source in "${jnu_sources[@]}"; do
    for target in "${jnu_sources[@]}"; do
      if [ "$source" != "$target" ]; then
        python train.py --model_name="$model" --source="$source" --target="$target" --Domain="JNU_imba" --imba True
      fi
    done
  done
done


## Cross-machine domain adaptation experiments

# # Calculate the total number of experiments
# total_experiments=$((${#models[@]} * (${#CWRU_sources[@]} * ${#jnu_sources[@]} + ${#jnu_sources[@]} * ${#CWRU_sources[@]})))
# current_experiment=1


# for model in "${models[@]}"; do
#     # Cross from CWRU to JNU
#     for source in "${CWRU_sources[@]}"; do
#         for target in "${jnu_sources[@]}"; do
#             echo "Training model $model from source $source to target $target (Experiment $current_experiment of $total_experiments)"
#             python train.py --model_name="$model" --source="$source" --target="$target" --Domain="Cross-machine"
#             ((current_experiment++))
#         done
#     done
#     # Cross from JNU to CWRU
#     for source in "${jnu_sources[@]}"; do
#         for target in "${CWRU_sources[@]}"; do
#             echo "Training model $model from source $source to target $target (Experiment $current_experiment of $total_experiments)"
#             python train.py --model_name="$model" --source="$source" --target="$target" --Domain="Cross-machine"
#             ((current_experiment++))
#         done
#     done
# done
