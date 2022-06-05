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
    # TODO improve?
    model_path = os.path.join(
        src_path,
        'mlruns', '4',
        '9bb8c4fed76e47c78c8c8c669e177318', 'artifacts',
        '22_06_04_00_59_38_transpose_all_chord_extended_7_batchsize_64_seed_1234567890.pt'
    )

    model = torch.load(open(model_path, 'rb'))
    model_class = type(model)

    metadata = 78  # TODO load from tempo mapping
    temperature = 0.01
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
