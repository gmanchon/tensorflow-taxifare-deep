
from tensorflow_taxifare_deep.trainer import Trainer


class TestTrainerTraining:

    def test_training(self):
        """
        run a training
        """
        # Act
        trainer = Trainer(nrows=100_000)
        trainer.clean()
        trainer.preproc(test_size=0.3)
        trainer.fit(plot_history=True, verbose=1)
        trainer.evaluate()
        trainer.save_model()

        # Assert
        # no assertion
