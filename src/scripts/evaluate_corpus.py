import os
import sys
from glob import glob
import logging

import numpy as np
import pandas as pd

from src.evaluation import evaluate_timestep_melody, evaluate_duration_melody, SEQUENCE_LENGTHS, EVALUATION_METRICS

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logFormatter = logging.Formatter('%(levelname)7s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

dir_path = os.path.dirname(os.path.realpath(__file__))
src_path = os.path.join(dir_path, '..', '..')


if __name__ == "__main__":
    metrics = {}
    encoding_type = 'duration'

    folder = os.path.join(src_path, 'data', 'encoded', encoding_type, 'mono')
    filepaths = [y for x in os.walk(folder) for y in glob(os.path.join(x[0], '*.csv'))]

    corpus_sequences = None

    # logger.info('Calculate corpus sequences')
    do_duration = False
    # corpus_sequences = {}
    # for l in SEQUENCE_LENGTHS:
    #     corpus_sequences[l] = {}
    #     for filepath in filepaths:
    #         df = pd.read_csv(filepath, index_col=0)
    #         df = df[df['type'] == 'improvised']
    #         df = df.dropna()
    #
    #         for i in range(df.shape[0] - l + 1):
    #             pitch = tuple(df.iloc[i:i + l, 2].astype(int).values)
    #
    #             if do_duration:
    #                 duration = tuple(df.iloc[i:i + l, 3].astype(int).values)
    #
    #                 seq = (pitch, duration)
    #             else:
    #                 seq = pitch
    #
    #             corpus_sequences[l][seq] = True

    logger.info('Calculate metrics')
    for filepath in filepaths:
        if encoding_type == 'timestep':
            metrics[filepath] = evaluate_timestep_melody(filepath, corpus_sequences)
        elif encoding_type == 'duration':
            metrics[filepath] = evaluate_duration_melody(filepath, corpus_sequences)

    metrics_df = pd.DataFrame().from_dict(metrics).T
    metrics_df['HC-m'] = metrics_df['HC'].apply(np.mean)

    for metric in EVALUATION_METRICS:
        if metric in metrics_df.columns:
            logger.info(f'{metric} - {metrics_df[metric].mean():5.3f} - {metrics_df[metric].std():5.3f}')
        else:
            logger.error(f'{metric} has not been calculated')
