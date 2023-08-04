"""
Executes the data loading and training of the model. The goal of this iteration is to predict both
the pulse height and photon arrival times using a common FC layer and common CNN for feature extraction.
The hope is that the predicitive power for both variables in one FC layer is possible. Pulse height
in this experiment is the NON-NORMALIZED phase response peaks.
"""

import mlflow
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader 
from mlcore.training import train_step, test_step
from mlcore.eval import regression_error
from mlcore.dataset import save_model, load_training_data, stream_to_height, stream_to_arrival

def main():
    # Define run hyperparams and constants
    NUM_SAMPLES = 1000
    WINDOW_SIZE = 1000
    EDGE_PAD = 10
    RANDOM_SEED = 42
    TEST_RATIO = 0.2
    BATCH_SIZE = 64
    LR = 0.01
    EPOCHS = 2000
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

    # Define dataset locations and load the training and test data
    test_dir = Path('../../../data/pulses/test/single_pulse/variable_qp_density/raw_iq')
    train_dir = Path('../../../data/pulses/train/single_pulse/variable_qp_density/raw_iq')
    fname = Path(f'vp_single_num{NUM_SAMPLES}_win{WINDOW_SIZE}_pad{EDGE_PAD}.npz')
    labels = ('i', 'q', 'photon_arrivals', 'phase_response')
    i_test, q_test, arrs_test, theta1_test = load_training_data(test_dir / fname, labels=labels)
    i_train, q_train, arrs_train, theta1_train = load_training_data(train_dir / fname, labels=labels)

    # Now we want to expand the dimensions for the i and q streams
    # since they will be used as input samples.
    i_test = np.expand_dims(i_test, axis=1)
    i_train = np.expand_dims(i_train, axis=1)
    q_test = np.expand_dims(q_test, axis=1)
    q_train = np.expand_dims(q_train, axis=1)

    # Get pulse heights and photon arrival values
    target_arrs_train = stream_to_arrival(arrs_train)
    target_arrs_test = stream_to_arrival(arrs_test)
    target_pulse_train = stream_to_height(arrs_train, theta1_train)
    target_pulse_test = stream_to_height(arrs_test, theta1_test)

    # Now we want to convert the loaded data to tensors.
    # Shape for targets is NUM_SAMPLES x 1 x 2
    # Shape for inputs is NUM_SAMPLES x 2 x WINDOW_SIZE
    X_train = torch.Tensor(np.hstack((i_train, q_train))) 
    X_test = torch.Tensor(np.hstack((i_test, q_test)))
    y_train = torch.Tensor(np.stack((target_arrs_train, target_pulse_train), axis=2)) 
    y_test = torch.Tensor(np.stack((target_arrs_test, target_pulse_test), axis=2))

    # Create the Dataloaders from newly created tensors
    train_dloader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dloader = DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

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
            mlflow.log_metric('train_error', train_metrics['acc'], epoch)
            mlflow.log_metric('test_error', test_metrics['acc'], epoch)

    # Save model weights for review
    save_model(MODEL_DIR, MODEL_FNAME, conv_reg_v2, 'pt')

if __name__ == '__main__':
   main()