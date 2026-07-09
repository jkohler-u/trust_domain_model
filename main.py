from pathlib import Path
import numpy as np
import pandas as pd
import pyddm
import pyddm.plot

import matplotlib.pyplot as plt

from utils.lossRL import LossRLIntegrated
from utils.rl_ddm import trust_drift, sim_rlddm_integrated

# To get a baseline, we query a participant about their general confidence
# The participant is then presented with a “speaker” who is an expert/not an expert in a domain
# Again, to get a baseline: For each of the (here two) speakers, the participant rates the perceived expertise of the speaker per domain
# Now for the main part of the experiment:
# Iteratively, the participant then sees statements by the “speaker”
# The participant then decides whether they trust that statement in a RT
# The statements can be correct/false, and the participant gets feedback on that (reward)
# The statements are within a domain (that can either be inside the domain of the speaker or outside the domain of the speaker)
# Afterwards, the participant once again rates the perceived expertise of the speaker per domain to get a baseline

# Reinforcement Learning Component:
# for domain:
# Qdomain1[trial] = Qdomain1[trial-1] + learning_rate * (true_reward - Qdomain1[trial-1])
# Qdomain2[trial] = Qdomain2[[trial-1] + learning_rate * (true_reward- Qdomain2[trial-1])
# overall: 
# Qoverall[trial] = Qoverall[[trial-1] + learning_rate * (true_reward - Qoverall[trial-1])


PLOT_DIR = Path(__file__).parent.parent / "plots"

if __name__ == "__main__":
    # Separate model for simulation
    model_sim = pyddm.gddm(
        drift=trust_drift,
        noise=1,
        bound=1,
        nondecision=0,
        conditions=["q_domain", "q_overall", "expertise"],
        parameters={
            "drift_intercept": 0, 
            "drift_scaling": 1, 
            "omega": 0.5, 
            "alpha": 0.1
        },
        T_dur=20
    )

    # 2. Parameters to fit
    # Note: we include alpha in the drift dependence so the Loss function can find it
    recov_params = {
        "drift_intercept": ( -1, 1),
        "drift_scaling": (0, 5),
        "omega": (0, 1),
        "alpha": (0, 1) 
    }

    # 3. Setup Model
    model_recov = pyddm.gddm(
        drift=trust_drift,
        noise=1,
        bound=1,
        nondecision=0,
        conditions=["q_domain", "q_overall", "expertise"],
        parameters=recov_params,
        T_dur=20
    )

    # 4. Simulate data
    # Use model_sim for the simulation, and model_recov for the fitting
    # Assume Speaker 1 is expert in A, novice in B
    expertise_map = {
        'Speaker1': {'A': 0.8, 'B': 0.2}, 
        'Speaker2': {'A': 0.3, 'B': 0.9}
    }
    sim_df = sim_rlddm_integrated(
        m_sim=model_sim, 
        n_trials=100, 
        n_sessions=2, 
        reward_probs=[0.7, 0.3], 
        alpha=0.1, 
        expertise_map=expertise_map
        )
    
    # Convert pandas to pyddm Sample
    samp = pyddm.Sample.from_pandas_dataframe(sim_df, choice_column_name="choice", rt_column_name="rt")

    # 5. Fit using the new Integrated Loss
    model_recov.fit(sample=samp, lossfunction=LossRLIntegrated)

    model_recov.parameters()