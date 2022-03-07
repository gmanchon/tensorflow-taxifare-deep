
test package for taxifare deep

added features:
- save sklearn pipeline and tensorflow model to gcp bucket
- download sklearn pipeline and tensorflow model from gcp bucket
- make prediction using downloaded model and pipeline
- prediction api

# tests

``` bash
pytest tests/test_training.py           # trigger training from cli
pytest tests/test_prediction.py         # trigger prediction from cli
```
