"""
Executes the data loading and training of the model. The goal of this iteration is to predict both
the pulse height and photon arrival times using a common FC layer and common CNN for feature extraction.
The hope is that the predicitive power for both variables in one FC layer is possible.
"""

import mlflow
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader, random_split
import random
from sklearn.model_selection import train_test_split

from mlcore.training import train_step, test_step
from mlcore.eval import regression_error
from mlcore.dataset import save_model

def main():
    # Define run hyperparams and constants
    NUM_SAMPLES = 1000
    WINDOW_SIZE = 1000
    EDGE_PAD = 60
    RANDOM_SEED = 42
    TEST_RATIO = 0.2
    BATCH_SIZE = 32
    LR = 0.01
    EPOCHS = 10000
    MODEL_DIR = Path.cwd() / 'trained_models'
    MODEL_FNAME = 'cnn_reg'

    # Define params to record in mlflow run
    params = {
       'num_samples': NUM_SAMPLES,
       'window_size': WINDOW_SIZE,
       'edge_pad': EDGE_PAD,
       'seed': RANDOM_SEED,
       'batch_size': BATCH_SIZE,
       'learning_rate': LR,
       'epoch_count': EPOCHS,
       'test_train_ratio': TEST_RATIO
    }

    # Enabling device agnostic code
    if torch.cuda.is_available():
      device = torch.device("cuda")

    elif torch.backends.mps.is_available():
      device = torch.device('mps')

    else:
      device = torch.device("cpu")
    print(f'Using device: "{device}"')

    # Load the appropriate dataset
    data_dir = '../../../data/pulses/single_pulse/'
    pulse_list = np.load(data_dir + f'vp_single_num{NUM_SAMPLES}_win{WINDOW_SIZE}_pad{EDGE_PAD}.npz')
    pulses = list(pulse_list['pulses'])

    # Split the data into the training samples and targets after shuffling the dataset.
    # Using the photon_arrival timestream to determine where to look in the qp density stream for the
    # pulse height value. Note, converting from ndarrays to tensors here.
    X, y = [], []
    random.shuffle(pulses)
    for element in pulses:
        X.append(element[0:2,:]) # I and Q timestreams
        arr_time = np.argwhere(element[2] == 1).item()
        pulse_height = element[3][arr_time].item()
        y.append(np.array([arr_time / WINDOW_SIZE, pulse_height]).reshape(1,2)) # Scaling arrival time [0, WINDOW_SIZE] -> [0, 1]


    # With the training and label data now separated, lets split the dataset into train and test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_RATIO, # Ratio of test data to use from full dataset; Training is the complement
        random_state=RANDOM_SEED)

    # Create Dataloader objects
    # Let's first convert from numpy arrays to Tensors and create datasets
    X_train = torch.Tensor(np.array(X_train))
    X_test = torch.Tensor(np.array(X_test))
    y_train = torch.Tensor(np.array(y_train))
    y_test = torch.Tensor(np.array(y_test))

    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)

    train_dloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    ## Model Definition and Training ##
    from models import ConvRegv2

    # Lets create a model instance, loss, and optimizer
    torch.manual_seed(RANDOM_SEED)
    conv_reg_v2 = ConvRegv2(2, BATCH_SIZE)
    optimizer = torch.optim.SGD(params=conv_reg_v2.parameters(), lr=LR)
    loss_fn = torch.nn.L1Loss(reduction='mean')

    # Define training/testing loops and log them to mlflow
    mlflow.set_tracking_uri(Path.cwd().parent / 'mlruns')
    mlflow.set_experiment('cnn_regression')
    with mlflow.start_run():
        mlflow.log_params(params)

        for epoch in range(EPOCHS):
            print(f'Epoch: {epoch + 1}/{EPOCHS}', end='\r')
            train_metrics = train_step(conv_reg_v2,
                                       train_dloader,
                                       loss_fn,
                                       optimizer,
                                       regression_error,
                                       device)

            test_metrics = test_step(conv_reg_v2,
                                     test_dloader,
                                     loss_fn,
                                     regression_error,
                                     device)

            # Log desired metrics for this iteration
            mlflow.log_metric('l1loss_train', train_metrics['loss'], epoch)
            mlflow.log_metric('l1loss_test', test_metrics['loss'], epoch)
            mlflow.log_metric('train_accuracy', train_metrics['acc'], epoch)
            mlflow.log_metric('test_accuracy', test_metrics['acc'], epoch)

    # Save model weights for review
    save_model(MODEL_DIR, MODEL_FNAME, conv_reg_v2, 'pt')

if __name__ == '__main__':
   main()