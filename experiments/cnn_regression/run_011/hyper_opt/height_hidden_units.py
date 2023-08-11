"""
This is an experiment to see how modulating the hidden unit count in the 
phase response peak FC network affects model performance.
"""

import mlflow
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import TensorDataset, DataLoader 
from mlcore.training import multi_loss_train_step, multi_loss_test_step
from mlcore.eval import regression_error
from mlcore.dataset import save_model, load_training_data, stream_to_arrival
from mlcore.models import BranchedConvReg

def main():
    # Define run hyperparams and constants
    NUM_SAMPLES = 20000
    WINDOW_SIZE = 1000
    EDGE_PAD = 10
    RANDOM_SEED = 22
    BATCH_SIZE = 256
    LR = 0.001
    EPOCHS = 5000
    MODEL_DIR = Path.cwd().parent / 'trained_models'
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
       'height_hidden_layers': 3,
       'mlflow_x_axis': 'height network hidden units'
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
    test_dir = Path('/home/kids/src/ml_readout/data/pulses/test/single_pulse/variable_qp_density/normalized_iq')
    train_dir = Path('/home/kids/src/ml_readout/data/pulses/train/single_pulse/variable_qp_density/normalized_iq')
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
    target_pulse_train = np.min(theta1_train, axis=1, keepdims=True)
    target_pulse_test = np.min(theta1_test, axis=1, keepdims=True)

    # Now we want to convert the loaded data to tensors.
    # Shape for targets is NUM_SAMPLES x 1 x 2
    # Shape for inputs is NUM_SAMPLES x 2 x WINDOW_SIZE
    X_train = torch.Tensor(np.hstack((i_train, q_train))) 
    X_test = torch.Tensor(np.hstack((i_test, q_test)))
    y_train = torch.Tensor(np.stack((target_arrs_train, target_pulse_train), axis=2)) 
    y_test = torch.Tensor(np.stack((target_arrs_test, target_pulse_test), axis=2))

    # Create the dataloaders from newly created tensors.
    train_dloader = DataLoader(dataset=TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    test_dloader = DataLoader(dataset=TensorDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

    # Now lets start the outer loop that will loop over all values in the
    # hidden units list.
    HYPERPARAM = [unit for unit in range(10, 105, 10)]
    mlflow.set_tracking_uri(Path.cwd().parent.parent / 'mlruns')
    mlflow.set_experiment('cnn_regression')
    with mlflow.start_run():
        mlflow.log_params(params)

        # Want to save the best overall model for further evaluation.
        # The figure of merit is the sum of the average test loss and 
        # average train loss for a given iteration (for pulse height only).
        fom = 0
        for idx, units in enumerate(HYPERPARAM):
            print(f'\nIteration: {idx + 1}/{len(HYPERPARAM)}')
            
            # Create container for values to report to mlflow.
            # Want to report the minimum and average train/test loss per iteration
            iter_train_loss = []
            iter_test_loss = []


            # Generate new model for each iteration, modulating
            # the hidden layer number in the height regression FC network.
            torch.manual_seed(RANDOM_SEED)
            model = BranchedConvReg(2, h_hidden_units=units, h_hidden_layers=3)
            optimizer = torch.optim.SGD(params=model.parameters(), lr=LR)
            height_loss_fn = torch.nn.L1Loss(reduction='mean')
            arrvival_loss_fn = torch.nn.L1Loss(reduction='mean')

            # Perform training/testing
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

                # Record desired metrics for this epoch
                iter_train_loss.append(train_metrics['1']['loss'])
                iter_test_loss.append(test_metrics['1']['loss'])

            # Convert to arrays to get aggregation functions
            iter_train_loss = np.array(iter_train_loss)
            iter_test_loss = np.array(iter_test_loss)

            # Save the model if it's the overall best
            if idx == 0:
               best_model = [model]
               fom = iter_train_loss.mean() + iter_test_loss.mean()
            elif iter_train_loss.mean() + iter_test_loss.mean() < fom:
               best_model = [model]

            # Report metrics to mlflow
            mlflow.log_metric('min_train_loss', iter_train_loss.min(), units)
            mlflow.log_metric('min_test_loss', iter_test_loss.min(), units)
            mlflow.log_metric('max_train_loss', iter_train_loss.max(), units)
            mlflow.log_metric('max_test_loss', iter_test_loss.max(), units)
            mlflow.log_metric('mean_train_loss', iter_train_loss.mean(), units)
            mlflow.log_metric('mean_test_loss', iter_test_loss.mean(), units)

    # Save the best model       
    save_model(MODEL_DIR, MODEL_FNAME, best_model[0], 'pt')

if __name__ == '__main__':
   main()