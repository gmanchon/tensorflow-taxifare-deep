
from tensorflow_taxifare_deep.trainer import Trainer


class TestTrainerPrediction:

    def test_prediction(self):
        """
        run a prediction
        """
        # Act
        trainer = Trainer(nrows=0)
        trainer.load_model()
        X_pred = trainer.get_X_pred_example()
        y_pred = trainer.predict(X_pred)

        # Assert
        # no assertion
