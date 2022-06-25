import os
import sys
import logging
import time

import torch
import mlflow
import yaml

from src.dataset import MelodyDataset

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logFormatter = logging.Formatter('%(levelname)7s - %(message)s')

consoleHandler = logging.StreamHandler(sys.stdout)
consoleHandler.setFormatter(logFormatter)
logger.addHandler(consoleHandler)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

dir_path = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(dir_path, '..', '..')


if __name__ == "__main__":
    tracking_uri = os.path.join(root_path, 'mlruns')
    mlflow.set_tracking_uri(tracking_uri)

    with open("../config/training_config.yml", "r") as stream:
        try:
            config = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            exit(1)

    mlflow_config = config['mlflow']
    experiment_name = mlflow_config['experiment_name']
    run_name = mlflow_config['run_name']

    dataset_config = config['dataset']
    encoding_type = dataset_config['encoding_type']
    polyphonic = bool(dataset_config['polyphonic'])
    chord_encoding_type = dataset_config['chord_encoding_type']
    chord_extension_count = int(dataset_config['chord_extension_count'])
    transpose_mode = dataset_config['transpose_mode']
    sequence_size = int(dataset_config['sequence_size'])

    model_config = config['model']
    use_padding_idx = bool(model_config['use_padding_idx'])
    start_pitch_symbol = int(model_config['start_pitch_symbol'])
    end_pitch_symbol = int(model_config['end_pitch_symbol'])
    start_attack_symbol = int(model_config['start_attack_symbol'])
    end_attack_symbol = int(model_config['end_attack_symbol'])
    start_duration_symbol = int(model_config['start_duration_symbol'])
    end_duration_symbol = int(model_config['end_duration_symbol'])

    offset_size = int(model_config['offset_size'])
    pitch_size = int(model_config['pitch_size'])
    attack_size = int(model_config['attack_size'])
    metadata_size = int(model_config['metadata_size'])

    embedding_size = int(model_config['embedding_size'])
    lstm_num_layers = int(model_config['lstm_num_layers'])
    lstm_hidden_size = int(model_config['lstm_hidden_size'])
    nn_hidden_size = int(model_config['nn_hidden_size'])
    nn_output_size = int(model_config['nn_output_size'])

    embedding_dropout_rate = float(model_config['embedding_dropout_rate'])
    lstm_dropout_rate = float(model_config['lstm_dropout_rate'])
    nn_dropout_rate = float(model_config['nn_dropout_rate'])

    normalize = bool(model_config['normalize'])
    gradient_clipping = float(model_config['gradient_clipping'])

    pitch_loss_weight = float(model_config['pitch_loss_weight'])
    attack_loss_weight = float(model_config['attack_loss_weight'])
    duration_loss_weight = float(model_config['duration_loss_weight'])

    training_config = config['training']
    num_batches = int(training_config['num_batches'])
    batch_size = int(training_config['batch_size'])
    num_epochs = int(training_config['num_epochs'])
    seed = int(training_config['seed'])
    learning_rate = float(training_config['learning_rate'])
    momentum = float(training_config['momentum'])
    weight_decay = float(training_config['weight_decay'])

    if chord_encoding_type == 'compressed' and chord_extension_count != 12:
        raise Exception(f"Chord extension count has to be 12 for encoding type 'compressed'")

    if encoding_type == 'timestep_base':
        from src.model import TimeStepBaseModel

        model_class = TimeStepBaseModel
    elif encoding_type == 'timestep_chord':
        from src.model import TimeStepChordModel

        model_class = TimeStepChordModel
    elif encoding_type == 'timestep':
        from src.model import TimeStepFullModel

        model_class = TimeStepFullModel
    elif encoding_type == 'duration_base':
        from src.model import DurationBaseModel

        model_class = DurationBaseModel
    elif encoding_type == 'duration_chord':
        from src.model import DurationChordModel

        model_class = DurationChordModel
    elif encoding_type == 'duration':
        from src.model import DurationFullModel

        model_class = DurationFullModel
    else:
        raise Exception(f'Unknown encoding type: {encoding_type}')

    try:
        logger.debug(f'Create new experiment: {experiment_name}')
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException:
        logger.debug(f'Experiment {experiment_name} already exists')
        experiment = mlflow.set_experiment(experiment_name)
        experiment_id = experiment.experiment_id

    mlflow.start_run(experiment_id=experiment_id, run_name=run_name)
    run = mlflow.active_run()
    run_id = run.info.run_id

    mlflow.log_param('encoding_type', encoding_type)
    mlflow.log_param('polyphonic', polyphonic)
    mlflow.log_param('chord_encoding_type', chord_encoding_type)
    mlflow.log_param('chord_extension_count', chord_extension_count)
    mlflow.log_param('transpose_mode', transpose_mode)
    mlflow.log_param('sequence_size', sequence_size)

    mlflow.log_param('use_padding_idx', use_padding_idx)
    mlflow.log_param('start_pitch_symbol', start_pitch_symbol)
    mlflow.log_param('end_pitch_symbol', end_pitch_symbol)
    mlflow.log_param('start_attack_symbol', start_attack_symbol)
    mlflow.log_param('end_attack_symbol', end_attack_symbol)
    mlflow.log_param('start_duration_symbol', start_duration_symbol)
    mlflow.log_param('end_duration_symbol', end_duration_symbol)
    mlflow.log_param('offset_size', offset_size)
    mlflow.log_param('pitch_size', pitch_size)
    mlflow.log_param('attack_size', attack_size)
    mlflow.log_param('metadata_size', metadata_size)
    mlflow.log_param('embedding_size', embedding_size)
    mlflow.log_param('embedding_dropout_rate', embedding_dropout_rate)
    mlflow.log_param('lstm_num_layers', lstm_num_layers)
    mlflow.log_param('lstm_hidden_size', lstm_hidden_size)
    mlflow.log_param('lstm_dropout_rate', lstm_dropout_rate)
    mlflow.log_param('nn_hidden_size', nn_hidden_size)
    mlflow.log_param('nn_output_size', nn_output_size)
    mlflow.log_param('nn_dropout_rate', nn_dropout_rate)
    mlflow.log_param('normalize', normalize)
    mlflow.log_param('gradient_clipping', gradient_clipping)
    mlflow.log_param('pitch_loss_weight', pitch_loss_weight)
    mlflow.log_param('attack_loss_weight', attack_loss_weight)
    mlflow.log_param('duration_loss_weight', duration_loss_weight)

    mlflow.log_param('num_batches', num_batches)
    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('seed', seed)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('momentum', momentum)
    mlflow.log_param('weight_decay', weight_decay)

    melody_dataset = MelodyDataset(
        encoding_type=encoding_type,
        polyphonic=polyphonic,
        chord_encoding_type=chord_encoding_type,
        chord_extension_count=chord_extension_count,
        transpose_mode=transpose_mode,
        logger=logger
    )
    melody_dataset.load()

    mlflow.log_param('dataset', melody_dataset.name)
    mlflow.log_param('num_examples', len(melody_dataset.tensor_dataset))

    model = model_class(
        dataset=melody_dataset,
        logger=logger,
        save_path=run.info.artifact_uri,
        sequence_size=sequence_size,
        use_padding_idx=use_padding_idx,
        start_pitch_symbol=start_pitch_symbol,
        end_pitch_symbol=end_pitch_symbol,
        start_attack_symbol=start_attack_symbol,
        end_attack_symbol=end_attack_symbol,
        start_duration_symbol=start_duration_symbol,
        end_duration_symbol=end_duration_symbol,
        offset_size=offset_size,
        pitch_size=pitch_size,
        attack_size=attack_size,
        metadata_size=metadata_size,
        embedding_size=embedding_size,
        embedding_dropout_rate=embedding_dropout_rate,
        lstm_num_layers=lstm_num_layers,
        lstm_hidden_size=lstm_hidden_size,
        lstm_dropout_rate=lstm_dropout_rate,
        nn_hidden_size=nn_hidden_size,
        nn_output_size=nn_output_size,
        nn_dropout_rate=nn_dropout_rate,
        normalize=normalize,
        gradient_clipping=gradient_clipping,
        pitch_loss_weight=pitch_loss_weight,
        attack_loss_weight=attack_loss_weight,
        duration_loss_weight=duration_loss_weight
    )

    optimizer = torch.optim.SGD(  # TODO try Adam
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[10, 20, 30],
        gamma=0.5
    )

    def callback(status, training_results):
        if len(training_results) > 0:
            for metric_name, metric_value in training_results.iloc[-1].iteritems():
                mlflow.log_metric(metric_name, metric_value)
        mlflow.end_run(status)
        time.sleep(10)

    completed = model.train_and_eval(
        num_batches=num_batches,
        batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        seed=seed,
        callback=callback
    )

    # if completed:
    #     temperature = 1.0
    #     sample = (False, False)
    #
    #     generator = model.GENERATOR_CLASS(
    #         model,
    #         temperature,
    #         sample,
    #         logger
    #     )
    #
    #     evaluate_model(model, generator, logger, n_measures=8)
