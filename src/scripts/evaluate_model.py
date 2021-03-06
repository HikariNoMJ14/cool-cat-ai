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
        src_path, 'mlruns',
        # time-step base
        # '11/029e40e890b847229c6419b7a40d8f89/artifacts',
        # '22_06_21_01_09_26_transpose_all_chord_extended_7_batchsize_128_seed_1234567890.pt'

        # time-step full
        # '10/f7a9e47e5b5b4e6f8177ddc52531b3cb/artifacts',
        # '22_06_24_20_26_34_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # '10/d7440c51ae474828b64d3e11c741c8e1/artifacts',
        # '22_06_25_23_32_27_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # duration base
        # '12/08a17afd032945acb22a30eb5aea8ab9/artifacts',
        # '22_06_21_21_48_06_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # '5/11d02c2e8c5a493cbf0fb33a787582f2/artifacts',
        # '22_06_28_01_59_32_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # duration chord
        # '13/d6c446b7dd8e47799a3ce0ae3cd6cecc/artifacts',
        # '22_06_22_14_35_54_transpose_all_chord_extended_7_batchsize_128_seed_1234567890_best_val.pt'

        # '5/7214bf20585d41f28cf09fbfc319aa6e/artifacts',
        # '22_06_27_21_23_31_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # duration full
        # '5/10890460b0ea43fea7e57354d0835405/artifacts',
        # '22_06_07_00_15_51_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'

        # '5/8df8f5afa93048cdae3606b065cd02fc/artifacts',
        # '22_06_24_01_24_37_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # '5/86bb55472c2f4797b00e486efe09fe15/artifacts',
        # '22_06_27_02_04_16_transpose_all_chord_extended_7_batchsize_128_seed_9876543_best_val.pt'

        # compressed chords
        '5/78dd3f9ddbd94933a0a8dc4c2cdc0cca/artifacts',
        '22_06_28_15_04_23_transpose_all_chord_compressed_12_batchsize_128_seed_9876543_best_val.pt'
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

    evaluate_model(model, generator, logger, n_samples=50)
