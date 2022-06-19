
from src.generator import TimeStepBaseGenerator


class TimeStepChordGenerator(TimeStepBaseGenerator):

    def __init__(self, model, metadata, temperature, sample, logger):
        super(TimeStepChordGenerator, self).__init__(model, metadata, temperature, sample, logger)

    def generate_melody(self, melody_name, n_measures):
        super().generate_melody(melody_name, n_measures)
