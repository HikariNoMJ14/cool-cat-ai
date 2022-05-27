import os
import sys
import logging

import torch

from src.generator import DurationGenerator

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
        'mlruns', '2',
        '603b3504919e46b3980a9ddfcc110fcf', 'artifacts',
        '22_05_22_00_52_58_transpose_all_chord_extended_7_batchsize_128_seed_1234567890.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    temperature = 0.01
    sample = (False, False)

    generator = DurationGenerator(
        model,
        temperature,
        sample,
        logger
    )

    generator.generate_melody('A Felicidade', 32)
    generator.save()
