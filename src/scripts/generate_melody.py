import os
import sys
import logging

import torch
import numpy as np

from src.model import DurationBaseModel, DurationChordModel, DurationFullModel, \
    TimeStepBaseModel, TimeStepChordModel, TimeStepFullModel
from src.generator import DurationBaseGenerator, DurationChordGenerator, DurationFullGenerator, \
    TimeStepBaseGenerator, TimeStepChordGenerator, TimeStepFullGenerator
from src.utils import tempo_to_metadata

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter('%(levelname)7s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')

seed = 666

if __name__ == "__main__":
    model_path = os.path.join(
        src_path, 'mlruns',

        # time-step base
        # '11/029e40e890b847229c6419b7a40d8f89', 'artifacts',
        # '22_06_21_01_09_26_transpose_all_chord_extended_7_batchsize_128_seed_1234567890.pt'

        # time-step full
        '10/f7a9e47e5b5b4e6f8177ddc52531b3cb/artifacts',
        '22_06_24_20_26_34_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # duration base
        # '12/08a17afd032945acb22a30eb5aea8ab9', 'artifacts',
        # '22_06_21_21_48_06_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration full
        # '5/10890460b0ea43fea7e57354d0835405', 'artifacts',
        # '22_06_07_00_15_51_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'
    )

    model = torch.load(open(model_path, 'rb'))
    model_class = type(model)

    tempo = 120
    metadata = tempo_to_metadata(tempo)
    temperature = 1
    sample = (False, False)

    if model_class == DurationBaseModel:
        generator_class = DurationBaseGenerator
    elif model_class == DurationChordModel:
        generator_class = DurationChordGenerator
    elif model_class == DurationFullModel:
        generator_class = DurationFullGenerator
    elif model_class == TimeStepBaseModel:
        generator_class = TimeStepBaseGenerator
    elif model_class == TimeStepChordModel:
        generator_class = TimeStepChordGenerator
    elif model_class == TimeStepFullModel:
        generator_class = TimeStepFullGenerator
    else:
        logger.error(f'Generator not found for {model_class}')
        exit(1)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    generator = generator_class(
        model,
        temperature,
        sample,
        logger
    )

    generator.generate_melody('Autumn Leaves', metadata, 32)
    generator.save(tempo=tempo)
