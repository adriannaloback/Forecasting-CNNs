__copyright__ = "Adrianna Loback 2019"

import os
import json
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt
from core.utils import Timer
from core.data_processor import DataLoader
from core.eval_metrics import EvalMetrics
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
#import pdb


def plot_results(predicted_data, true_data):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.ylabel('Close (GBPUSD)')
    plt.xlabel('Time (Hours)')
    plt.show()


def plot_results_multiple(predicted_data, true_data, prediction_len):
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
	# Pad the list of predictions to shift it in the graph to its correct start
    for i, data in enumerate(predicted_data):
        padding = [None for p in range(i * prediction_len)]
        plt.plot(padding + data, label='Prediction')
        plt.legend()
    plt.ylabel('Close (GBPUSD)')
    plt.show()


def denorm_transform(p0_vec, n_pred, n_true):
    p_pred = np.multiply(p0_vec, (n_pred+1))
    p_true = np.multiply(p0_vec, (n_true+1))
    return p_pred,p_true


def main():
    configs = json.load(open('config.json', 'r'))
    if not os.path.exists(configs['model']['save_dir']): os.makedirs(configs['model']['save_dir'])
    
    # -- Data preparation: --
    data = DataLoader(
        os.path.join('../data', configs['data']['filename']),
        configs['data']['train_test_split'],
        configs['data']['columns'])

    x, y = data.get_train_data(
           seq_len=configs['data']['sequence_length'],
           normalise=configs['data']['normalise'])

    x_test, y_test, p0_vec = data.get_test_data(
           seq_len=configs['data']['sequence_length'],
           normalise=configs['data']['normalise'])

    # -- Init and fit CNN model: --
    n_features = x.shape[2]
    n_steps = configs['data']['sequence_length'] - 1
    
    # Define model
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='linear', input_shape=(n_steps, n_features)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(50, activation='linear'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    # Compile model
    model.compile(optimizer=configs['model']['optimizer'], loss=configs['model']['loss'])
    
    # Fit model
    timer = Timer()
    timer.start()
    print('[Model] Training Started')
    model.fit(x, y, epochs=configs['training']['epochs'], batch_size=configs['training']['batch_size'])
    timer.stop()
    print('[Model] Predicting one step ahead...')

    # Get predictions
    yhat = model.predict(x_test, verbose=0)

    # Denormalize & plot 
    p_pred, p_true = denorm_transform(p0_vec, yhat, y_test)
    plot_results(p_pred, p_true) #de-normalised, i.e., original fex units

    # Compute evaluation metrics
    assess = EvalMetrics(p_true, p_pred)
    MAE    = assess.get_MAE()
    RMSE   = assess.get_RMSE()
    print("MAE on validation set is: %f" % MAE)
    print("RMSE on validation set is: %f" % RMSE)
    
    # Save model
    save_dir = configs['model']['save_dir']
    save_fname = os.path.join(save_dir, '%s_cnn.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S')))
    model.save(save_fname)

if __name__ == '__main__':
    main()
