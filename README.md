
based on the [taxi-fare-deep model](https://github.com/lewagon/taxi-fare-deep) and used by the [lw vertex api tutorial](https://kitt.lewagon.com/knowledge/tutorials/vertex_api)

test package for taxifare deep

added features:
- save sklearn pipeline and tensorflow model to gcp bucket
- download sklearn pipeline and tensorflow model from gcp bucket
- make prediction using downloaded model and pipeline
- prediction api
- used in order to run the tutorial on production API with GPU using Vertex AI Workbench

# tests

``` bash
pytest tests/test_training.py           # trigger training from cli
pytest tests/test_prediction.py         # trigger prediction from cli
```

# container

## apple silicon

``` bash
make run_api                            # test api locally

make build_m1_local                     # build image locally
make run_m1_local                       # verify image
make build_m1_prod                      # build image for prod

make push_image                         # push image
make deploy_image                       # deploy image
```

## other machines

``` bash
make run_api                            # test api locally

make build_image                        # build image locally
make run_image                          # verify image

make push_image                         # push image
make deploy_image                       # deploy image
```
