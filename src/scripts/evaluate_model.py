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
        src_path,
        'mlruns', '5',
        '72b4d266dbca40308624362723ce7f11', 'artifacts',
        '22_06_01_23_14_12_transpose_all_chord_extended_7_batchsize_64_seed_1234567890_best_val.pt'
    )

    model = torch.load(open(model_path, 'rb'))

    evaluate_model(model, logger)