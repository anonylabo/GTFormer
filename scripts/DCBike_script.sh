#!/bin/bash

MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

for model in ${MODEL[@]}
do
python main.py --model $model --city DC --data_type Bike --save_outputs True
done
