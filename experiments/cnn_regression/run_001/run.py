"""
Executes the data loading and training of the model
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from timeit import default_timer as timer
import random
from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split

from mlcore.training import train_step, test_step, make_predictions
from mlcore.eval import accuracy_regression, plot_stream_data

def main():
    # Define run hyperparams
    NUM_SAMPLES = 1000
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

    # Load the appropriate dataset
    data_dir = '../../../data/pulses/single_pulse/'
    pulse_list = np.load(data_dir + f'p_single_num{NUM_SAMPLES}_win{WINDOW_SIZE}_pad{EDGE_PAD}.npz')
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
    loss_fn = torch.nn.L1Loss(reduction='mean')

    # Define training/testing loops
    total_time = lambda start_time, stop_time: stop_time - start_time
    train_time_cnn_start = timer()
    for epoch in tqdm(range(EPOCHS)):
        print(f'Epoch: {epoch}\n-----------')
        train_step(
            conv_reg_v1,
            train_dloader,
            loss_fn,
            optimizer,
            accuracy_regression,
            device
        )
        test_step(
            conv_reg_v1,
            test_dloader,
            loss_fn,
            accuracy_regression,
            device
        )
    train_time_cnn_end = timer()
    print(f'Total time to train: {total_time(train_time_cnn_start, train_time_cnn_end):.2f}s')   
    
if __name__ == '__main__':
   main()
        