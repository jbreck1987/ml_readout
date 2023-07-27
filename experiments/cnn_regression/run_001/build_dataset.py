"""
Generates source data for this particular run if it doesn't already exist.
"""

import numpy as np
from pathlib import Path

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream as QPT
from mlcore.dataset import make_dataset

def main():
    # Define data storage parent location
    datadir = '../../../data/pulses/'


    # Define data variables
    NUM_PULSE_FRACTION = 0.0
    NUM_SAMPLES = [x for x in range(100, 1100, 100)]
    QPT_TIMELEN = 0.01 # Secs
    SAMPLING_FREQ = 2e6 # Hz
    FALL_TIME = 30 # uSecs
    WINDOW_SIZE = 1000 # uSecs
    SINGLE_PULSE = True # Only want data with one pulse per window in this run
    EDGE_PAD = [x for x in range(10, 110, 10)] # Padding at beginning and end of samples where no pulse is allowed
    
    # Create temp containers for pulse data before writing
    pulses = []
    no_pulses = []

    # Define Quasiparticle Timestream object and run the loop through all the iterations of the data above
    qpt = QPT(SAMPLING_FREQ, QPT_TIMELEN)
    for num_samples in NUM_SAMPLES:
        for pad in EDGE_PAD:
            if Path(datadir, f'p_single_num{num_samples}_win{WINDOW_SIZE}_pad{pad}.npz').exists():
                print('passing, file exists...')
            print(f'Generating file: "p_single_num{num_samples}_win{WINDOW_SIZE}_pad{pad}.npz"')

if __name__ == '__main__':
    main()