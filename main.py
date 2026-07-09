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

# After each trial (per domain)l:
#   If person acts trustworthy/untrustworthy:
#     Qdomain ← Qdomain + α × (R - Qdomain) 
#     Qoverall ← Qoverall + α × (R - Qoverall)

# Where (per domain):
# α = learning rate (0 to 1)
# R = reward you received (1 for correct, -1 for incorrect) → whether the participant's response aligned with their choice

# drift_rate = drift_rate_intercept + drift_rate_scaling x …. (function of q-values)
# drift_rate = omega(perceived_domain_expertise_speaker *  softmax(Qdomain)) + (1-omega) softmax(Qoverall )
# => how would we solve that the Q values are on different scales 

# Each trial/ DDM:
# z/bias: perceived_domain_expertise_speaker of the speaker
# a/caution: general trustingness of the participant
# drift rate: predicted by the RL → Q
#  t0: non-decision-time - fixed across participants (?)
# => output: decision of the participant / RT of the participant 


PLOT_DIR = Path(__file__).parent.parent / "plots"

if __name__ == "__main__":
    # Create model object with fixed parameters to simulate data

    # Create solution object (to later simulate data; represents probability distribution function over time for choices associated with upper and lower bound crossings)

    # Create model to fit object for parameter recovery

    # visualize model to fit object using model GUI

    # use solution object to simulate data

    # fit the model to the simulated data using the LossRLFast loss function

    
    ...