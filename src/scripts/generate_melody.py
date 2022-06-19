import os
import sys
import logging

import torch

from src.model import DurationBaseModel, DurationChordModel, DurationFullModel, \
    TimeStepBaseModel, TimeStepChordModel, TimeStepFullModel
from src.generator import DurationBaseGenerator, DurationChordGenerator, DurationFullGenerator, \
    TimeStepBaseGenerator, TimeStepChordGenerator, TimeStepFullGenerator

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter('%(levelname)7s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


if __name__ == "__main__":
    model_path = os.path.join(
        src_path, 'mlruns',
        '8/a8e1970e4f354cc782fc6ae2491bb305', 'artifacts',
        '22_06_18_00_22_20_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'
    )

    model = torch.load(open(model_path, 'rb'))
    model_class = type(model)

    metadata = 78  # TODO load from tempo mapping
    temperature = 1.0
    sample = (False, False)

    if model_class == DurationBaseModel:
        generator_class = DurationBaseGenerator
    elif model_class == DurationChordModel:
        generator_class = DurationChordGenerator
    elif model_class == DurationFullModel:
        generator_class = DurationFullGenerator
    if model_class == TimeStepBaseModel:
        generator_class = TimeStepBaseGenerator
    elif model_class == TimeStepChordModel:
        generator_class = TimeStepChordGenerator
    elif model_class == TimeStepFullModel:
        generator_class = TimeStepFullGenerator
    else:
        logger.error(f'Generator not found for {model_class}')
        exit(1)

    generator = generator_class(
        model,
        metadata,
        temperature,
        sample,
        logger
    )

    generator.generate_melody('A Felicidade', 32)
    generator.save()
