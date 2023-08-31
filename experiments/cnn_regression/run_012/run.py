"""
Executes the data loading and training of the model. The goal of this iteration is to predict both
the pulse height and photon arrival times using a common FC layer and common CNN for feature extraction.
The hope is that the predicitive power for both variables in one FC layer is possible. Pulse height
in this experiment is the non-normalized phase response peaks. Lowered noise scale to make improve achievable R.
"""

import mlflow
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader 
from mlcore.training import multi_loss_train_step, multi_loss_test_step, EarlyStop
from mlcore.eval import regression_error
from mlcore.dataset import save_model, load_training_data, stream_to_arrival

def main():
    # Define run hyperparams and constants
    NUM_SAMPLES = 20000
    WINDOW_SIZE = 1000
    EDGE_PAD = 10
    RANDOM_SEED = 22
    TEST_RATIO = 0.2
    BATCH_SIZE = 64 
    LR = 0.001
    EPOCHS = 20000
    MODEL_DIR = Path.cwd() / 'trained_models'
    MODEL_FNAME = 'cnn_reg'
    H_UNITS = 100
    H_LAYERS = 3
    NOISE_SCALE = 30

    # Define params to record in mlflow run
    params = {
       'num_samples': NUM_SAMPLES,
       'window_size': WINDOW_SIZE,
       'edge_pad': EDGE_PAD,
       'seed': RANDOM_SEED,
       'batch_size': BATCH_SIZE,
       'learning_rate': LR,
       'epoch_count': EPOCHS,
       'test_train_ratio': TEST_RATIO,
       'height_hidden_units': H_UNITS,
       'height_hidden_layers': H_LAYERS,
       'white_noise_scale': NOISE_SCALE
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
    test_dir = Path('../../../data/pulses/test/single_pulse/variable_qp_density/normalized_iq')
    train_dir = Path('../../../data/pulses/train/single_pulse/variable_qp_density/normalized_iq')
    fname = Path(f'vp_single_num{NUM_SAMPLES}_win{WINDOW_SIZE}_noisescale{NOISE_SCALE}_pad{EDGE_PAD}.npz')
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
    target_pulse_train = np.min(theta1_train, axis=1, keepdims=True)
    target_pulse_test = np.min(theta1_test, axis=1, keepdims=True)

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
    from models import BranchedConvReg

    # Lets create a model instance, loss, and optimizer
    torch.manual_seed(RANDOM_SEED)
    model = BranchedConvReg(2, h_hidden_units=H_UNITS, h_hidden_layers=H_LAYERS)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
    height_loss_fn = torch.nn.L1Loss(reduction='mean')
    arrvival_loss_fn = torch.nn.L1Loss(reduction='mean')

    # Define training/testing loops and log them to mlflow.
    # Early stop is used to increase efficiency by stopping the run if it stalls
    # or has divergence between the test/train metrics. For this run, stalling in the
    # test loss for the height regression network will be monitored.
    early_stop = EarlyStop(sat_tolerance=int(EPOCHS * 0.02), sat_metric=1, track_sat=True)

    mlflow.set_tracking_uri(Path.cwd().parent / 'mlruns')
    mlflow.set_experiment('cnn_regression')
    with mlflow.start_run():
        mlflow.log_params(params)

        for epoch in range(EPOCHS):
            print(f'Epoch: {epoch + 1}/{EPOCHS}', end='\r')
            train_metrics = multi_loss_train_step(model,
                                                  train_dloader,
                                                  (arrvival_loss_fn, height_loss_fn),
                                                  optimizer,
                                                  regression_error,
                                                  device)

            test_metrics = multi_loss_test_step(model,
                                                test_dloader,
                                                (arrvival_loss_fn, height_loss_fn),
                                                regression_error,
                                                device)

            # Log desired metrics for this iteration
            mlflow.log_metric('l1loss_train_arrival', train_metrics['0']['loss'], epoch)
            mlflow.log_metric('l1loss_train_height', train_metrics['1']['loss'], epoch)
            mlflow.log_metric('l1loss_test_arrival', test_metrics['0']['loss'], epoch)
            mlflow.log_metric('l1loss_test_height', test_metrics['1']['loss'], epoch)

            # Will stop the run early based on the height regression metrics
            if early_stop(train_metrics['1']['loss'], test_metrics['1']['loss']):
               print(f'Stopped early in Epoch {epoch}')
               break

    # Save model weights for review
    save_model(MODEL_DIR, MODEL_FNAME, model, 'pt')

if __name__ == '__main__':
   main()
