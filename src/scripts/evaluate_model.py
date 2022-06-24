import os
import sys
import logging

import torch

from src.evaluation import evaluate_model

logger = logging.getLogger()
logger.setLevel(logging.INFO)
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
        # src_path, 'mlruns',
        # '10/ff85d1433722430caf4acad32e48e80c', 'artifacts',
        # '22_06_22_23_33_22_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration base
        # src_path, 'mlruns',
        # '12/08a17afd032945acb22a30eb5aea8ab9', 'artifacts',
        # '22_06_21_21_48_06_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration chord
        # src_path, 'mlruns',
        # '13/d6c446b7dd8e47799a3ce0ae3cd6cecc', 'artifacts',
        # '22_06_22_14_35_54_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # duration full
        src_path, 'mlruns',
        # '5/10890460b0ea43fea7e57354d0835405/artifacts',
        # '22_06_07_00_15_51_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    temperature = 1
    sample = (False, False)

    generator = model.GENERATOR_CLASS(
        model,
        temperature,
        sample,
        logger
    )

    evaluate_model(model, generator, logger, n_samples=100)

