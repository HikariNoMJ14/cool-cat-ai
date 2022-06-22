import os
import sys
import logging

import torch

from src.evaluation import evaluate_model

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
        # time-step base
        # src_path, 'mlruns',
        # '11/029e40e890b847229c6419b7a40d8f89', 'artifacts',
        # '22_06_21_01_09_26_transpose_all_chord_extended_7_batchsize_128_seed_1234567890.pt'

        # time-step full
        src_path, 'mlruns',
        '10/f67d6b65ec6846b8971352b7367ab6d4', 'artifacts',
        '22_06_20_12_03_43_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration base
        # src_path, 'mlruns',
        # '12/08a17afd032945acb22a30eb5aea8ab9', 'artifacts',
        # '22_06_21_21_48_06_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration full
        # src_path, 'mlruns',
        # '5/10890460b0ea43fea7e57354d0835405', 'artifacts',
        # '22_06_07_00_15_51_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    metadata = 78  # TODO load from tempo mapping
    temperature = 1.0
    sample = (False, False)

    generator = model.GENERATOR_CLASS(
        model,
        temperature,
        sample,
        logger
    )

    evaluate_model(model, generator, logger)

