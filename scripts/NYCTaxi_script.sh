#!/bin/bash

MODEL=("AR" "LSTM" "GEML" "CrowdNet" "GTFormer")

for model in ${MODEL[@]}
do
python main.py --model $model --data_type Taxi --save_outputs True
done
