"""
Generates source data for this particular run if it doesn't already exist.
"""

import numpy as np
import os
from math import ceil
from pathlib import Path

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream as QPT
from mlcore.dataset import make_dataset, save_training_data

def main():
    # Define data storage parent location
    data_parent_dir = os.environ['ML_DATA_DIR']
    data_dir = Path(data_parent_dir + '/pulses/test/single_pulse/variable_qp_density/normalized_iq')

    # Define lambda function that determines number of samples based on
    # sampling rate and how long of a window (in time) is desired
    samples = lambda samp_rate, interval: ceil(samp_rate * (interval * 1e-6)) # interval is uSecs

    # Define data variables
    NO_PULSE_FRACTION = 0
    NUM_SAMPLES = 30000
    MAGS = [800/x for x in range(1100, 1400, 100)]
    QPT_TIMELEN = 0.01 # Secs
    SAMPLING_FREQ = 1e6 # Hz
    WINDOW_SIZE = samples(SAMPLING_FREQ, 1000) # Samples
    SINGLE_PULSE = True # Only want data with one pulse per window in this run
    EDGE_PAD = 100 # Padding at beginning and end of samples where no pulse is allowed
    CPS = 500 # Photon arrival rate

    # Define Quasiparticle Timestream object and run the loop through all the iterations of the parameters
    # above to generate the dataset
    qpt = QPT(SAMPLING_FREQ, QPT_TIMELEN)

    for mag in MAGS:
        mag_str =f'{mag:.3f}'.replace('.', '')
        p = Path(data_dir, f'vp_single_num{NUM_SAMPLES}_window{WINDOW_SIZE}_mag{mag_str}_nonoise_pad{EDGE_PAD}')
        if p.exists():
            print('passing, file exists...')
            
        else:
            # Create temp containers for pulse data before writing
            pulses = []
            no_pulses = []
            
            # Generate the training/test/validation samples
            make_dataset(
                qpt,
                [mag],
                NUM_SAMPLES,
                NO_PULSE_FRACTION,
                pulses,
                no_pulses,
                SINGLE_PULSE,
                False,
                CPS,
                EDGE_PAD,
                WINDOW_SIZE)

            # Save data to disk
            save_training_data(pulses, data_dir, p.stem)

if __name__ == '__main__':
    main()
