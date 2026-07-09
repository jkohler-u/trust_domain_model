from pathlib import Path
import numpy as np
import pandas as pd
import pyddm

import matplotlib.pyplot as plt

from utils.lossRL import LossRL, LossRLFast
from utils.rl_ddm import sim_rlddm, m_sim

# NOTE:
# The RL-DDM depends on “deltaq” as a condition, even though this does not (and cannot) exist in the data. 
# This is because we set it in the loss function as we iterate through the trials, running an RL model with the given parameters. 
# In practice, this means that you do not have to explicitly pass deltaq as a condition when fitting, 
# because the loss function calls the model with the appropriate deltaq. 
# However, when simulating trials or viewing the model using the model GUI, you must provide a deltaq.

PLOT_DIR = Path(__file__).parent.parent / "plots"

if __name__ == "__main__":
    # Simulate data
    m_sim = pyddm.gddm(
        drift=lambda deltaq, driftscale : driftscale * deltaq,
        noise=1,
        bound=1,
        nondecision=0,
        parameters={"driftscale": 1},
        conditions=["deltaq"], T_dur=10
    )

    samp = sim_rlddm(m_sim, 1000, 2, [.2, .8], .1)

    # Build and fit the model
    m = pyddm.gddm(drift=lambda driftscale,deltaq,alpha : driftscale * deltaq, # Hack including alpha
               noise=1,
               bound=1,
               nondecision=0,
               T_dur=20, dx=.01, dt=.01,
               conditions=["deltaq", "session", "trial", "reward"],
               parameters={"driftscale": (0, 8), "alpha": (0, 1)})

    # Fit the model
    m.fit(sample=samp, lossfunction=LossRLFast)

    # To get the qleft and qright values for the last session, run this:
    pyddm.get_model_loss(model=m, sample=samp, lossfunction=LossRLFast)
    qleft = m.last_qleft
    qright = m.last_qright

    pyddm.plot.model_gui(m, conditions={"deltaq": [-1, -.8, -.6, -.4, -.2, 0, .2, .4, .6, .8, 1]})

    # see the parameters after fitting
    m.parameters()
