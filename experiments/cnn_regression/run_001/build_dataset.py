"""
Generates source data for this particular run if it doesn't already exist.
"""

import numpy as np
from pathlib import Path

from mkidreadoutanalysis.quasiparticletimestream import QuasiparticleTimeStream as QPT
from mlcore.dataset import make_dataset

def main():
    # Define data storage parent location
    datadir = '../../../data/pulses/single_pulse/'


    # Define data variables
    NO_PULSE_FRACTION = 0
    NUM_SAMPLES = [x for x in range(100, 1100, 100)]
    QPT_TIMELEN = 0.01 # Secs
    SAMPLING_FREQ = 2e6 # Hz
    FALL_TIME = 30 # uSecs
    WINDOW_SIZE = 1000 # uSecs
    SINGLE_PULSE = True # Only want data with one pulse per window in this run
    EDGE_PAD = [x for x in range(10, 110, 10)] # Padding at beginning and end of samples where no pulse is allowed
    CPS = 1000 # Photon arrival rate

    # Define Quasiparticle Timestream object and run the loop through all the iterations of the parameters
    # above to generate the dataset
    qpt = QPT(SAMPLING_FREQ, QPT_TIMELEN)
    qpt.gen_quasiparticle_pulse(tf = FALL_TIME)

    for num_samples in NUM_SAMPLES:
        for pad in EDGE_PAD:
            p = Path(datadir, f'p_single_num{num_samples}_win{WINDOW_SIZE}_pad{pad}.npz')
            if p.exists():
                print('passing, file exists...')
                
            else:
                # Create temp containers for pulse data before writing
                pulses = []
                no_pulses = []
                
                # Generate the training/test/validation samples
                make_dataset(
                    qpt,
                    num_samples,
                    NO_PULSE_FRACTION,
                    pulses,
                    no_pulses,
                    SINGLE_PULSE,
                    CPS,
                    pad,
                    WINDOW_SIZE)

                # Convert to numpy array and save to disk as npz
                np.savez(p, pulses=np.array(pulses))

if __name__ == '__main__':
    main()