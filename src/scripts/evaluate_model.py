import os
import sys
import logging

import torch

from src.melody import Melody
from src.generator import DurationGenerator
from src.utils import get_original_filepath, get_filepaths

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
        'mlruns', '2',
        '603b3504919e46b3980a9ddfcc110fcf', 'artifacts',
        '22_05_22_00_52_58_transpose_all_chord_extended_7_batchsize_128_seed_1234567890.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    temperature = .999
    sample = (False, False)

    generator = DurationGenerator(
        model,
        temperature,
        sample,
        logger
    )

    original_filepaths = set([Melody(i, '1.2').song_name for i in get_filepaths('original')])
    seen_filepaths = set([Melody(i, '1.2').song_name for i in get_filepaths('improvised')])
    unseen_filepaths = original_filepaths.difference(seen_filepaths)

    print(f'S: {seen_filepaths}')
    print(f'U: {unseen_filepaths}')

    # seen_gen_filepaths = []
    # for filepath in seen_filepaths:
    #     print(f"Seen: {filepath}")
    #     generator.generate_melody(filepath, 32)
    #
    #     seen_gen_filepaths.append(
    #         generator.save('seen')
    #     )

    unseen_gen_filepaths = []
    for filepath in unseen_filepaths:
        print(f"Unseen: {filepath}")
        generator.generate_melody(filepath, 32)
        unseen_gen_filepaths.append(
            generator.save('unseen')
        )
