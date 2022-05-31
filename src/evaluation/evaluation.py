import numpy as np
import pandas as pd

from src.melody import Melody
from src.generator import DurationGenerator
from src.utils import get_filepaths
from src.evaluation import objective_metrics

EVALUATION_METRICS = ['PCHE1', 'PCHE4', 'PVF4', 'TS12', 'CPR2', 'DPR12', 'GPS', 'RV4', 'QR', 'HC-m']


def evaluate_melody(filepath):
    df = pd.read_csv(filepath, index_col=0)

    df['measure'] = (df['ticks'] // 48).astype('int')
    df = df.rename({
        'improvised_pitch': 'pitch',
        'improvised_duration': 'duration'
    }, axis=1)
    df['offset'] = df['offset'].astype('int')
    df['ticks'] = df['ticks'].astype('int')
    df['duration'] = df['duration'].astype('int')
    df['pitch'] = df['pitch'].replace(np.nan, -1).astype('int')

    # PCHE
    pe1 = objective_metrics.compute_piece_pitch_entropy(df, 1)
    pe4 = objective_metrics.compute_piece_pitch_entropy(df, 4)
    # PVF
    pvf4 = objective_metrics.calculate_PVF(df[df['pitch'] >= 0], 4)
    # TS
    ts12 = objective_metrics.calculate_TS(df, 12)
    # CPR
    cpr2 = objective_metrics.calculate_CPR(df, 2)
    # DPR
    dpr12 = objective_metrics.calculate_DPR(df, 12)
    # GPS
    gs = objective_metrics.compute_piece_groove_similarity(df, max_pairs=np.inf)
    # RV
    rv4 = objective_metrics.calculate_RV(df, 4)
    # QR
    qr = objective_metrics.calculate_QD(df)
    # HC
    hc = objective_metrics.calculate_HC(df[df['pitch'] >= 0])
    # RN
    #     rm = objective_metrics.calculate_RM(df)

    return {
        'PCHE1': pe1, 'PCHE4': pe4, 'PVF4': pvf4, 'TS12': ts12,
        'CPR2': cpr2, 'DPR12': dpr12, 'GPS': gs, 'RV4': rv4,
        'QR': qr, 'HC': hc,
        #         'RM': rm
    }


def evaluate_model(model, logger, unseen=False):
    metrics = {}
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

    logger.debug(f'S: {seen_filepaths}')
    logger.debug(f'U: {unseen_filepaths}')

    seen_gen_filepaths = []
    for filepath in seen_filepaths:
        logger.debug(f"Seen: {filepath}")
        generator.generate_melody(filepath, 32)

        seen_gen_filepaths.append(
            generator.save('seen')
        )

    logger.info('Seen melodies')
    for generated_filepath in seen_gen_filepaths:
        metrics[generated_filepath] = evaluate_melody(generated_filepath)

    metrics_df = pd.DataFrame().from_dict(metrics).T
    metrics_df['HC-m'] = metrics_df['HC'].apply(np.mean)

    for metric in EVALUATION_METRICS:
        logger.info(f'{metric} - {metrics_df[metric].mean():5.2f} - {metrics_df[metric].std():5.2f}')

    if unseen:
        unseen_gen_filepaths = []
        for filepath in unseen_filepaths:
            logger.debug(f"Unseen: {filepath}")
            generator.generate_melody(filepath, 32)
            unseen_gen_filepaths.append(
                generator.save('unseen')
            )

            logger.info('Unseen melodies')
            for generated_filepath in unseen_gen_filepaths:
                metrics[generated_filepath] = evaluate_melody(generated_filepath)

            metrics_df = pd.DataFrame().from_dict(metrics).T
            metrics_df['HC-m'] = metrics_df['HC'].apply(np.mean)

            for metric in EVALUATION_METRICS:
                logger.info(f'{metric} - {metrics_df[metric].mean():5.2f} - {metrics_df[metric].std():5.2f}')