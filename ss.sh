#!/bin/bash

## cross-domain adaptation experiments

python train.py --model_name="proposed" --random_state 128
python train.py --model_name="proposed" --random_state 256
python train.py --model_name="proposed" --random_state 512
python train.py --model_name="proposed" --random_state 1024
python train.py --model_name="proposed" --random_state 2048





