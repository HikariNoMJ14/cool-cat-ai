import os
import sys
import logging

import torch

from src.model.duration import DurationBaseModel, DurationChordModel, DurationFullModel
from src.generator import DurationBaseGenerator, DurationChordGenerator, DurationFullGenerator

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
        src_path,
        'mlruns', '5',
        '10890460b0ea43fea7e57354d0835405', 'artifacts',
        '22_06_07_00_15_51_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'
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
