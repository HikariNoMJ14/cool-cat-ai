import os.path
import numpy as np
import pandas as pd

from src.melody import Melody
from src.utils import get_filepaths, random_metadata
from src.evaluation import objective_metrics

SEQUENCE_LENGTHS = [3, 4, 5, 6]
EVALUATION_METRICS = ['PCHE1', 'PCHE4', 'PVF4', 'TS12', 'CPR2', 'DPR24', 'GPS', 'RVF4', 'QRF', 'HC-m']

for l in SEQUENCE_LENGTHS:
    EVALUATION_METRICS.append(f'RMF{l}')


def calculate_metrics(df, corpus_sequences):
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
    dpr24 = objective_metrics.calculate_DPR(df, 24)
    # GPS
    gs = objective_metrics.compute_piece_groove_similarity(df, max_pairs=np.inf)
    # RVF
    rv4 = objective_metrics.calculate_RVF(df, 4)
    # QR
    qr = objective_metrics.calculate_QD(df)
    # HC
    hc = objective_metrics.calculate_HC(df[df['pitch'] >= 0])

    results = {
        'PCHE1': pe1, 'PCHE4': pe4, 'PVF4': pvf4, 'TS12': ts12,
        'CPR2': cpr2, 'DPR24': dpr24, 'GPS': gs, 'RVF4': rv4,
        'QRF': qr, 'HC': hc
    }

    # RMF
    if corpus_sequences is not None:
        for l in SEQUENCE_LENGTHS:
            rm = objective_metrics.calculate_RMF(
                df[df['pitch'] >= 0],
                corpus_sequences=corpus_sequences,
                l=l
            )
            results[f'RMF{l}'] = rm

    return results


def prepare_timestep_melody(filepath):
    df = pd.read_csv(filepath, index_col=0)

    df['measure'] = (df['ticks'] // 48).astype('int')
    df = df.rename({
        'improvised_pitch': 'pitch',
        'improvised_duration': 'duration'
    }, axis=1)
    df['offset'] = df['offset'].astype('int')
    df['ticks'] = df['ticks'].astype('int')
    # df['duration'] = df['duration'].astype('int')
    df['pitch'] = df['pitch'].replace(np.nan, -1).astype('int')

    return df


def evaluate_timestep_melody(filepath, corpus_sequences=None):
    df = prepare_timestep_melody(filepath)

    results = calculate_metrics(df, corpus_sequences)

    return results


def prepare_duration_melody(filepath):
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

    return df


def evaluate_duration_melody(filepath, corpus_sequences=None):
    df = prepare_duration_melody(filepath)

    if 'type' in df.columns:
        df = df[df['type'] == 'improvised']

    results = calculate_metrics(df, corpus_sequences)

    return results


def evaluate_model(model, generator, logger, n_measures=8, unseen=False, n_samples=100):
    metrics = {}

    corpus_sequences = None
    # corpus_sequences = {}
    # for l in SEQUENCE_LENGTHS:
    #     corpus_sequences[l] = objective_metrics.calculate_corpus_sequences(model.dataset, l)

    original_filepaths = set([Melody(i, '1.2').song_name for i in get_filepaths('original')])
    seen_filepaths = set([Melody(i, '1.2').song_name for i in get_filepaths('improvised')])
    unseen_filepaths = original_filepaths.difference(seen_filepaths)

    logger.debug(f'S: {seen_filepaths}')
    logger.debug(f'U: {unseen_filepaths}')

    seen_gen_filepaths = []

    for i in range(n_samples):
        filepath_idx = int(np.floor(np.random.random() * len(seen_filepaths)))
        filepath = list(seen_filepaths)[filepath_idx]
        tempo, metadata = random_metadata()

        logger.debug(f"Seen: {filepath}, tempo {tempo}")
        generator.generate_melody(filepath, metadata, n_measures)

        seen_gen_filepath = generator.save(tempo=tempo, save_path='seen')
        seen_gen_filepaths.append(seen_gen_filepath)

    logger.info('Seen melodies')
    for generated_filepath in seen_gen_filepaths:
        if model.ENCODING_TYPE == 'timestep':
            metrics[generated_filepath] = evaluate_timestep_melody(generated_filepath, corpus_sequences)
        elif model.ENCODING_TYPE == 'duration':
            metrics[generated_filepath] = evaluate_duration_melody(generated_filepath, corpus_sequences)

    metrics_df = pd.DataFrame().from_dict(metrics).T
    metrics_df['HC-m'] = metrics_df['HC'].apply(np.mean)

    metrics_df.to_csv(os.path.join(os.path.dirname(model.best_model_path), 'evaluation_seen.csv'))

    for metric in EVALUATION_METRICS:
        if metric in metrics_df.columns:
            logger.info(f'{metric} - {metrics_df[metric].mean():5.3f} - {metrics_df[metric].std():5.3f}')
        else:
            logger.error(f'{metric} has not been calculated')

    if unseen:
        unseen_gen_filepaths = []

        for i in range(n_samples):
            filepath_idx = np.random.random(len(seen_filepaths))
            filepath = list(seen_filepaths)[filepath_idx]
            tempo, metadata = random_metadata()

            logger.debug(f"Unseen: {filepath}")
            generator.generate_melody(filepath, metadata, n_measures)

            unseen_gen_filepath = generator.save(tempo=tempo, save_path='unseen')
            unseen_gen_filepaths.append(
                unseen_gen_filepath
            )

        logger.info('Unseen melodies')
        for generated_filepath in unseen_gen_filepaths:
            if model.ENCODING_TYPE == 'timestep':
                metrics[generated_filepath] = evaluate_timestep_melody(generated_filepath, corpus_sequences)
            elif model.ENCODING_TYPE == 'duration':
                metrics[generated_filepath] = evaluate_duration_melody(generated_filepath, corpus_sequences)

        metrics_df = pd.DataFrame().from_dict(metrics).T
        metrics_df['HC-m'] = metrics_df['HC'].apply(np.mean)

        metrics_df.to_csv(os.path.join(os.path.dirname(model.best_model_path), 'evaluation_unseen.csv'))

        for metric in EVALUATION_METRICS:
            if metric in metrics_df.columns:
                logger.info(f'{metric} - {metrics_df[metric].mean():5.3f} - {metrics_df[metric].std():5.3f}')
            else:
                logger.error(f'{metric} has not been calculated')
