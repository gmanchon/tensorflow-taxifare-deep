
import tensorflow as tf

from datetime import datetime
import pytz

import pandas as pd

# import joblib

from tensorflow_taxifare_deep.trainer import Trainer

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# http://127.0.0.1:8000/predict?pickup_datetime=2012-10-06 12:10:20&pickup_longitude=40.7614327&pickup_latitude=-73.9798156&dropoff_longitude=40.6513111&dropoff_latitude=-73.8803331&passenger_count=2


@app.get("/")
def index():
    return {
        "physical devices": tf.config.list_physical_devices("GPU"),
        "is GPU available": tf.test.is_gpu_available(),
        "GPU device name": tf.test.gpu_device_name()
    }


@app.get("/predict")
def predict(pickup_datetime: datetime,  # 2013-07-06 17:18:00
            pickup_longitude: float,    # -73.950655
            pickup_latitude: float,     # 40.783282
            dropoff_longitude: float,   # -73.984365
            dropoff_latitude: float,    # 40.769802
            passenger_count: int):      # 1
    """
    we use type hinting to indicate the data types expected
    for the parameters of the function
    FastAPI uses this information in order to hand errors
    to the developpers providing incompatible parameters
    FastAPI also provides variables of the expected data type to use
    without type hinting we need to manually convert
    the parameters of the functions which are all received as strings
    """

    # localize the user provided datetime with the NYC timezone
    eastern = pytz.timezone("US/Eastern")
    localized_pickup_datetime = eastern.localize(pickup_datetime, is_dst=None)

    # convert the user datetime to UTC
    utc_pickup_datetime = localized_pickup_datetime.astimezone(pytz.utc)

    # format the datetime as expected by the pipeline
    formatted_pickup_datetime = utc_pickup_datetime.strftime("%Y-%m-%d %H:%M:%S UTC")

    # fixing a value for the key, unused by the model
    # in the future the key might be removed from the pipeline input
    # eventhough it is used as a parameter for the Kaggle submission
    key = "2013-07-06 17:18:00.000000119"

    # build X ⚠️ beware to the order of the parameters ⚠️
    X_pred = pd.DataFrame(dict(
        key=[key],  # useless but the pipeline requires it
        pickup_datetime=[formatted_pickup_datetime],
        pickup_longitude=[pickup_longitude],
        pickup_latitude=[pickup_latitude],
        dropoff_longitude=[dropoff_longitude],
        dropoff_latitude=[dropoff_latitude],
        passenger_count=[passenger_count]))

    # pipeline = get_model_from_gcp()
    # pipeline = joblib.load('model.joblib')

    # step 3 : make prediction
    # y_pred = pipeline.predict(X_pred)

    # step 3 : make prediction
    trainer = Trainer(nrows=0)
    trainer.load_model()
    y_pred = trainer.predict(X_pred)

    # step 4 : return the prediction (extract from numpy array)
    pred = float(y_pred[0])

    return dict(prediction=pred)
