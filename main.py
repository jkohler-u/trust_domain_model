from pathlib import Path
import numpy as np
import pandas as pd
import pyddm
import pyddm.plot

import matplotlib.pyplot as plt

from utils.dataset import generate_ground_truth
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
    # --- CONFIGURATION ---
    speakers = ['Speaker1', 'Speaker2']
    domains = ['A', 'B']

    # ADDED: How often the speaker is actually right (Ground Truth)
    truth_probs = {
        'Speaker1': {'A': 0.8, 'B': 0.2}, 
        'Speaker2': {'A': 0.3, 'B': 0.9}
    }

    # How the participant perceives the speaker's expertise (Covariate)
    exp_map = {
        'Speaker1': {'A': 0.9, 'B': 0.1}, 
        'Speaker2': {'A': 0.2, 'B': 0.8}
    }

    # 1. Generate the "Script"
    gt_df = generate_ground_truth(
        n_sessions=2, 
        n_trials_per_speaker=50, 
        speakers=speakers, 
        domains=domains, 
        truth_probs=truth_probs
    )

    # 2. Simulate the Participant
    # Separate model for simulation
    model_sim = pyddm.gddm(
        drift=trust_drift,
        noise=1, # amount of random "jitter" / inconsistency of participant
        bound=1, # distance between yes/no boundaries
        nondecision=0,
        conditions=["q_domain", "q_overall", "expertise"], # need expertise in trust_drift, do not want column to be dropped
        parameters={
            "drift_intercept": 0, # general tendency to trust or distrust.
            "drift_scaling": 1, # confidence: how much do trust values influence behaviour
            "omega": 0.5, # weighting: omega determines tendency to trust domain-expertise over general trust
            "alpha": 0.1 # learning rate, value the Optimizer will try to find
        },
        T_dur=20 # maximum amount of time allowed to make decision (in s)
    )

    sim_df = sim_rlddm_integrated(
        model_sim, 
        gt_df, 
        alpha=0.1, # learning rate again, GT value
        expertise_map=exp_map # how participant rated speaker's domain expertise at start
    )

    # 2. Parameters to fit
    # Note: include alpha in drift dependence so Loss function finds it
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
    
    # Convert pandas to pyddm Sample
    samp = pyddm.Sample.from_pandas_dataframe(sim_df, choice_column_name="choice", rt_column_name="rt")

    # 5. Fit using the new Integrated Loss
    model_recov.fit(sample=samp, lossfunction=LossRLIntegrated)

    model_recov.parameters()