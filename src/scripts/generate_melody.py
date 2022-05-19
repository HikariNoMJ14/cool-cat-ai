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
        'f9f65390db154400be658fccefa23d45', 'artifacts',
        'transpose_all_chord_extended_7_batchsize_64_seed_1234567890.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    temperature = .99

    generator = DurationGenerator(
        model,
        temperature,
        logger
    )

    generator.generate_melody('A Felicidade', 32)
    generator.save()
