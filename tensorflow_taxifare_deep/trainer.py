from tensorflow_taxifare_deep.data import get_data, clean_data
from tensorflow_taxifare_deep.utils import plot_model_history, simple_time_tracker
from tensorflow_taxifare_deep.preprocessor import create_pipeline
from tensorflow_taxifare_deep.network import Network
from tensorflow_taxifare_deep.gcp_bucket import download_blob, upload_file

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import os

import joblib
from tensorflow.keras import models


class Trainer:
    def __init__(
            self,
            nrows=10_000,
            model_filename='my_model.h5',
            pipe_filename='my_pipeline.joblib'):

        self.df = get_data(nrows=nrows)
        self.X = None
        self.y = None
        self.X_train = None
        self.X_train_preproc = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.pipe = None
        self.network = None
        self.history = None

        # model local storage
        self.project_path = os.path.dirname(os.path.dirname(__file__))
        self.model_path = os.path.join(self.project_path, model_filename)
        self.pipe_path = os.path.join(self.project_path, pipe_filename)

        # model bucket storage
        self.bucket_name = 'le-wagon-ds'
        self.bucket_model_root = 'models/tensorflow_taxifare_deep'
        self.model_blob = f'{self.bucket_model_root}/{model_filename}'
        self.pipe_blob = f'{self.bucket_model_root}/{pipe_filename}'

    @simple_time_tracker
    def clean(self):
        print("###### loading and cleaning....")
        self.df = clean_data(self.df)
        self.X = self.df.drop("fare_amount", axis=1)
        self.y = self.df["fare_amount"]

    @simple_time_tracker
    def preproc(self, test_size=0.3):
        print("###### preprocessing....")
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=test_size
        )
        self.pipe = create_pipeline()
        self.X_train_preproc = self.pipe.fit_transform(self.X_train)
        print(
            "###### shape of X_train_preproc, y_train: ",
            self.X_train_preproc.shape,
            self.y_train.shape,
        )

    @simple_time_tracker
    def fit(self, plot_history=True, verbose=1):
        print("###### fitting...")
        # TensorFlow cannot work with Sparse Matrix out of Sklearn's OHE
        self.X_train_preproc = self.X_train_preproc.todense()
        self.network = Network(input_dim=self.X_train_preproc.shape[1])
        print(self.network.model.summary())
        self.network.compile_model()
        self.history = self.network.fit_model(
            self.X_train_preproc, self.y_train, verbose=verbose
        )

        # Print & plot some key training results
        print("####### min val MAE", min(self.history.history["val_mae"]))
        print("####### epochs reached", len(self.history.epoch))
        if plot_history:
            plot_model_history(self.history)

    @simple_time_tracker
    def evaluate(self, X_test=None, y_test=None):
        """evaluates the pipeline on a test set and return the MAE"""
        # If no test set is given, use the holdout from train/test/split
        print("###### evaluates the model on a test set...")
        X_test = X_test or self.X_test
        y_test = y_test or self.y_test
        y_test_pred = self.network.model.predict(self.pipe.transform(X_test))

        print("###### test score (MAE)", mean_absolute_error(y_test, y_test_pred))
        # todo

    @simple_time_tracker
    def save_model(self, model_filename=None, pipe_filename=None):

        # get model path
        if model_filename is None:
            model_filename = self.model_path
        if pipe_filename is None:
            pipe_filename = self.pipe_path

        # save model
        self.network.model.save(model_filename)
        joblib.dump(self.pipe, pipe_filename)

        # upload model to gcp
        upload_file(self.bucket_name, self.model_path, self.model_blob)
        upload_file(self.bucket_name, self.pipe_path, self.pipe_blob)

    @simple_time_tracker
    def load_model(self, model_filename=None, pipe_filename=None):

        # get model path
        if model_filename is None:
            model_filename = self.model_path
        if pipe_filename is None:
            pipe_filename = self.pipe_path

        # download model from gcp
        download_blob(self.bucket_name, self.model_blob, self.model_path)
        download_blob(self.bucket_name, self.pipe_blob, self.pipe_path)

        # load model
        self.loaded_model = models.load_model(model_filename)
        self.loaded_pipe = joblib.load(pipe_filename)

    @simple_time_tracker
    def get_X_pred_example(self):

        import pandas as pd

        # eventhough it is used as a parameter for the Kaggle submission
        key = "2013-07-06 17:18:00.000000119"

        # build X ⚠️ beware to the order of the parameters ⚠️
        X_pred = pd.DataFrame(dict(
            key=[key],  # useless but the pipeline requires it
            pickup_datetime=["2013-07-06 17:18:00 UTC"],
            pickup_longitude=[float("-73.950655")],
            pickup_latitude=[float("40.783282")],
            dropoff_longitude=[float("-73.984365")],
            dropoff_latitude=[float("40.769802")],
            passenger_count=[int("1")]))

        return X_pred

    @simple_time_tracker
    def predict(self, X_pred=None):
        print("###### predict...")
        assert(X_pred is not None)
        y_pred = self.loaded_model.predict(self.loaded_pipe.transform(X_pred))

        print(f'prediction 🌴 {y_pred}')

        return y_pred


if __name__ == "__main__":

    # Instanciate trainer with number of rows to download and use
    trainer = Trainer(nrows=5_000)

    # clean data
    trainer.clean()

    # Preprocess data and create train/test/split
    trainer.preproc(test_size=0.3)

    # Fit neural network and show training performance
    trainer.fit(plot_history=True, verbose=1)

    # evaluate on test set (by default the holdout from train/test/split)
    trainer.evaluate(X_test=None, y_test=None)
