import os
import sys
import logging

import torch
import mlflow
import yaml

from src.dataset import Dataset
from src.model import MonoTimeStepModel

logger = logging.getLogger()
logger.setLevel(logging.INFO)
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

    training_config = config['training']
    batch_size = int(training_config['batch_size'])
    num_epochs = int(training_config['num_epochs'])
    seed = int(training_config['seed'])
    learning_rate = float(training_config['learning_rate'])
    momentum = float(training_config['momentum'])
    weight_decay = float(training_config['weight_decay'])

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

    mlflow.log_param('batch_size', batch_size)
    mlflow.log_param('num_epochs', num_epochs)
    mlflow.log_param('seed', seed)
    mlflow.log_param('learning_rate', learning_rate)
    mlflow.log_param('momentum', momentum)
    mlflow.log_param('weight_decay', weight_decay)

    dataset = Dataset(
        encoding_type=encoding_type,
        polyphonic=polyphonic,
        chord_encoding_type=chord_encoding_type,
        chord_extension_count=chord_extension_count,
        transpose_mode=transpose_mode,
        sequence_size=sequence_size
    )
    dataset.load()

    print(dataset.tensor_dataset)

    mlflow.log_param('dataset', dataset.name)
    mlflow.log_param('num_examples', len(dataset.tensor_dataset))

    model = MonoTimeStepModel(
        dataset=dataset,
        logger=logger,
        save_path=run.info.artifact_uri,
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
        attack_loss_weight=attack_loss_weight
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[300, 400, 450],
        gamma=0.5
    )

    def callback(status, training_results):
        if len(training_results) > 0:
            for metric_name, metric_value in training_results.iloc[-1].iteritems():
                mlflow.log_metric(metric_name, metric_value)
        mlflow.end_run(status)

    model.train_and_eval(
        batch_size=batch_size,
        num_epochs=num_epochs,
        optimizer=optimizer,
        scheduler=scheduler,
        seed=seed,
        callback=callback
    )
