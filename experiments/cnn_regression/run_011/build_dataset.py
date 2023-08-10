"""
Generates source data for this particular run if it doesn't already exist.
"""

import numpy as np
from math import ceil
from pathlib import Path

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream as QPT
from mlcore.dataset import make_dataset, save_training_data

def main():
    # Define data storage parent location
    datadir = Path('../../../junk')

    # Define lambda function that determines number of samples based on
    # sampling rate and how long of a window (in time) is desired
    samples = lambda samp_rate, interval: ceil(samp_rate * (interval * 1e-6)) # interval is uSecs

    # Define data variables
    NO_PULSE_FRACTION = 0
    NUM_SAMPLES = [1000]
    MAGS = [1]
    QPT_TIMELEN = 0.01 # Secs
    SAMPLING_FREQ = 1e6 # Hz
    FALL_TIME = 30 # uSecs
    WINDOW_SIZE = samples(SAMPLING_FREQ, 1000) # Samples
    SINGLE_PULSE = True # Only want data with one pulse per window in this run
    EDGE_PAD = [10] # Padding at beginning and end of samples where no pulse is allowed
    CPS = 500 # Photon arrival rate

    # Define Quasiparticle Timestream object and run the loop through all the iterations of the parameters
    # above to generate the dataset
    qpt = QPT(SAMPLING_FREQ, QPT_TIMELEN)

    for num_samples in NUM_SAMPLES:
        for pad in EDGE_PAD:
            p = Path(datadir, f'verify_qpd1_num1000_window1000_nonorm_pad10')
            if p.exists():
                print('passing, file exists...')
                
            else:
                # Create temp containers for pulse data before writing
                pulses = []
                no_pulses = []
                
                # Generate the training/test/validation samples
                make_dataset(
                    qpt,
                    MAGS,
                    num_samples,
                    NO_PULSE_FRACTION,
                    pulses,
                    no_pulses,
                    SINGLE_PULSE,
                    CPS,
                    pad,
                    WINDOW_SIZE,
                    False)

                # Save data to disk
                save_training_data(pulses, datadir, p.stem)

if __name__ == '__main__':
    main()
