"""
Executes the data loading and training of the model. This run will
create simple loss curves for the data. We want to see how the loss
changes as a function of the number of training samples used with the
goal of seeing if the model is under/overfitting.
"""

import mlflow
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

from mlcore.training import train_step, test_step
from mlcore.eval import accuracy_regression

def main():
    # Define run hyperparams
    WINDOW_SIZE = 1000
    EDGE_PAD = 60
    RANDOM_SEED = 42
    TEST_RATIO = 0.2
    BATCH_SIZE = 32
    LR = 0.1
    EPOCHS = 100
     
    # Enabling device agnostic code
    if torch.cuda.is_available():
      device = torch.device("cuda")
    
    elif torch.backends.mps.is_available():
      device = torch.device('mps')
    
    else:
      device = torch.device("cpu")
    print(f'Using device: "{device}"')   
    
    # Define params to record in mlflow run
    params = {
       'window_size': WINDOW_SIZE,
       'edge_pad': EDGE_PAD,
       'seed': RANDOM_SEED,
       'batch_size': BATCH_SIZE,
       'learning_rate': LR,
       'epoch_count': EPOCHS,
       'test_train_ratio': TEST_RATIO
    }

    # Set up mlflow run
    mlflow.set_tracking_uri(Path.cwd().parent / 'mlruns') # Get the parent directory of the current working directory and append mlruns suffix
    mlflow.set_experiment('cnn_regression')
    with mlflow.start_run():
        mlflow.log_params(params)
      
        # Start loop over training sample number range
        start, end, step = 100, 1100, 100
        for id, num_samples in enumerate(range(start, end, step)):
            print(f'Iteration {id + 1}/{(end - step) / start}, Training samples: {num_samples}') # Console status udpates

            # Load the appropriate dataset
            data_dir = '../../../data/pulses/single_pulse/'
            pulse_list = np.load(data_dir + f'p_single_num{num_samples}_win{WINDOW_SIZE}_pad{EDGE_PAD}.npz')
            pulses = list(pulse_list['pulses'])

            # Split the data into the training samples and associated label
            X = []
            y = []

            # Lets create one big list of the pulse and no pulse samples randomly shuffled together 
            random.shuffle(pulses)

            # Now lets separate the training samples (I/Q data) from the label data (photon arrival element)
            # Note that for the label, feature scaling is being performed to scale the range: [0, 1000] -> [0,1]
            for element in pulses:
                X.append(element[0:2,:])
                y.append(np.argwhere(element[2] == 1) / WINDOW_SIZE)

            # With the training and label data now separated, lets split the dataset into train and test
            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_RATIO, # Ratio of test data to use from full dataset; Training is the complement
                random_state=RANDOM_SEED)

            # Now lets convert the lists to Tensors. Converting to np arrays first based on warning from torch
            X_train = torch.Tensor(np.array(X_train))
            X_test = torch.Tensor(np.array(X_test))
            y_train = torch.Tensor(np.array(y_train))
            y_test = torch.Tensor(np.array(y_test))

            # Create Dataloader objects
            # Let's first convert from numpy arrays to Tensors and create datasets
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)

            train_dloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
            test_dloader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False)

            ## Model Definition and Training ##
            from models import ConvRegv1

            # Lets create a model instance, loss, and optimizer
            torch.manual_seed(RANDOM_SEED)
            conv_reg_v1 = ConvRegv1(in_channels=2)
            optimizer = torch.optim.SGD(params=conv_reg_v1.parameters(), lr=LR)
            loss_fn = torch.nn.MSELoss(reduction='mean')

            # We'll be taking the average metric over all epochs and reporting
            # that to mlflow for each value of training sample quantity. Lets
            # create lists to hold the intermediate values
            train_loss = []
            test_loss = []
            train_acc = []
            test_acc = []

            # Define training/testing loops
            for _ in range(EPOCHS):
                train_metrics = train_step(conv_reg_v1,
                                           train_dloader,
                                           loss_fn,
                                           optimizer,
                                           accuracy_regression,
                                           device)

                test_metrics = test_step(conv_reg_v1,
                                         test_dloader,
                                         loss_fn,
                                         accuracy_regression,
                                         device)
                train_loss.append(train_metrics['loss'])
                test_loss.append(test_metrics['loss'])
                train_acc.append(train_metrics['acc'])
                test_acc.append(test_metrics['acc'])

            # Log desired metrics for this iteration
            mlflow.log_metric('mseloss_train', np.array(train_loss).mean(), num_samples)
            mlflow.log_metric('mseloss_test', np.array(test_loss).mean(), num_samples)
            mlflow.log_metric('train_acc', np.array(train_acc).mean(), num_samples)
            mlflow.log_metric('test_acc', np.array(test_acc).mean(), num_samples)

if __name__ == '__main__':
     main()
