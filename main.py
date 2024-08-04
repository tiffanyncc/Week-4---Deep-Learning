import os

# Set the environment variable to avoid the memory leak warning
os.environ["OMP_NUM_THREADS"] = "1"

import logging
import matplotlib.pyplot as plt
from src.data.make_dataset import load_data
from src.features.build_features import preprocess_data, prepare_data
from src.models.train import (
    train_neural_network, train_with_more_epochs, train_with_extra_layer, 
    train_with_more_neurons, train_with_new_learning_rate, 
    train_with_activation_function
)
from src.models.evaluate import find_best_learning_rate_with_scheduler
from src.visualization.visualize import plot_loss_curves, plot_learning_rate_vs_loss

# Setup logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

def main():
    try:
        df = load_data('src/data/employee_attrition.csv')
        logging.info('Data loaded successfully.')
    except Exception as e:
        logging.error(f'Error loading data: {e}')
        return

    try:
        df, X_scaled, Y = preprocess_data(df)
        logging.info('Data preprocessed successfully.')
    except Exception as e:
        logging.error(f'Error preprocessing data: {e}')
        return

    try:
        x_train, x_test, y_train, y_test = prepare_data(X_scaled, Y)
        logging.info('Data split and prepared successfully.')
    except Exception as e:
        logging.error(f'Error splitting and preparing data: {e}')
        return

    try:
        model, history, result = train_neural_network(x_train, y_train)
        logging.info('Neural Network trained successfully.')
    except Exception as e:
        logging.error(f'Error training neural network: {e}')
        return

    try:
        model_more_epochs, history_more_epochs, result_more_epochs = train_with_more_epochs(x_train, y_train)
        logging.info('Model with more epochs trained successfully.')
    except Exception as e:
        logging.error(f'Error training model with more epochs: {e}')

    try:
        model_extra_layer, history_extra_layer, result_extra_layer = train_with_extra_layer(x_train, y_train)
        logging.info('Model with extra layer trained successfully.')
    except Exception as e:
        logging.error(f'Error training model with extra layer: {e}')

    try:
        model_more_neurons, history_more_neurons, result_more_neurons = train_with_more_neurons(x_train, y_train)
        logging.info('Model with more neurons trained successfully.')
    except Exception as e:
        logging.error(f'Error training model with more neurons: {e}')

    try:
        model_new_lr, history_new_lr, result_new_lf = train_with_new_learning_rate(x_train, y_train)
        logging.info('Model with new learning rate trained successfully.')
    except Exception as e:
        logging.error(f'Error training model with new learning rate: {e}')

    try:
        lrs_scheduler, lr_scheduler_history = find_best_learning_rate_with_scheduler(x_train, y_train)
        plot_learning_rate_vs_loss(lrs_scheduler, lr_scheduler_history, 'learning_rate_vs_loss_scheduler')
        logging.info('Learning rate vs. loss with scheduler plot displayed and saved.')
    except Exception as e:
        logging.error(f'Error displaying learning rate vs. loss with scheduler plot: {e}')

    try:
        model_activation, history_activation, result_activation = train_with_activation_function(x_train, y_train)
        plot_loss_curves(history_activation, 'loss_curves_activation')
        logging.info('Model with activation function trained successfully.')
    except Exception as e:
        logging.error(f'Error training model with activation function: {e}')

if __name__ == '__main__':
    main()
