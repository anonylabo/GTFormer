# GTFormer: A Geospatial Temporal Transformer for Crowd Flow Prediction

This repository is the official implementation of "GTFormer: A Geospatial Temporal Transformer for Crowd Flow Prediction". 

<div align="center">
<img src="https://github.com/anonylabo/GTFormer/blob/main/figure/GTFormer.png" width="1000" alt="Figure" title="Architecture of GTFormer">
</div>


## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Experiment

1. Download the data from the following URL and place it in the ./data/DC_Bike, ./data/NYC_Bike and ./data/NYC_Taxi.


2. We provide the experiment scripts of all models under the folder ./scripts. You can reproduce the experiment results by:
   ```
   ./scripts/NYCBike_script.sh
   ./scripts/NYCTaxi_script.sh
   ./scripts/DCBike_script.sh
   ``` 


## Acknowledgement

We appreciate the following github repo a lot for their valuable code base

https://github.com/thuml/Autoformer

https://github.com/jonpappalord/crowd_flow_prediction

The data is provided by:

https://capitalbikeshare.com/system-data

https://citibikenyc.com/system-data

https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page
