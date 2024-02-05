#!/bin/bash

MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

for model in ${MODEL[@]}
do
python main.py --model $model --save_outputs True
done
