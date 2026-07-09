from pathlib import Path
import numpy as np
import pandas as pd
import pyddm
import pyddm.plot

import matplotlib.pyplot as plt

from utils.lossRL import LossRL, LossRLFast
from utils.rl_ddm import sim_rlddm, m_sim

# Reinforcement Learning Component:
# for domain:
# Qdomain1[trial] = Qdomain1[trial-1] + learning_rate * (true_reward - Qdomain1[trial-1])
# Qdomain2[trial] = Qdomain2[[trial-1] + learning_rate * (true_reward- Qdomain2[trial-1])
# overall: 
# Qoverall[trial] = Qoverall[[trial-1] + learning_rate * (true_reward - Qoverall[trial-1])


# drift_rate = drift_rate_intercept + drift_rate_scaling x …. (function of q-values)
# drift_rate = omega(perceived_domain_expertise_speaker *  softmax(Qdomain)) + (1-omega) softmax(Qoverall )
# -->
# drift_rate = drift_rate_intercept + drift_rate_scaling * (omega(perceived_domain_expertise_speaker * softmax(Q_domain) + (1-omega) * softmax(Q_overall)))???
# TODO: do some simulations, see whether Q values are on different scales 

    
# TODO: map our thoughts to the code in lossRL.py and rl_ddm.py, then to example code in rl_ddm_example.py


PLOT_DIR = Path(__file__).parent.parent / "plots"

if __name__ == "__main__":
    # TODO: map our thoughts to parameter values
    # --- RL parameters ---
    # update Q-values after each trial (per domain and overall)
    #     Qdomain ← Qdomain + α × (R - Qdomain) 
    #     Qoverall ← Qoverall + α × (R - Qoverall)
    # given:
        # alpha (value per participant): learning rate of the participant (0 to 1)
        # reward (value per trial): whether the participant's response aligned with their choice (1 if correct, -1 if incorrect)
    # to be fitted:
        # drift rate scaling (value per participant): how much the Q-values influence the drift rate
    # output:
        # Q-values for each domain and overall
    # --- DDM parameters ---
    # given: 
        # z/bias (value per domain per speaker): perceived_domain_expertise_speaker of the speaker
        # a/caution (value per participant): inverse confidence of the participant
        # t0: non-decision-time (fixed value per or across participants)
    # to be fitted:
        # drift rate (predicted by the RL -> Q)
    # output:
        # decision of the participant
        # RT of participant


    # Create model object with fixed parameters to simulate data

    # Create solution object (to later simulate data; represents probability distribution function over time for choices associated with upper and lower bound crossings)

    # Create model to fit object for parameter recovery

    # visualize model to fit object using model GUI

    # use solution object to simulate data

    # fit the model to the simulated data using the LossRLFast loss function

    
    ...