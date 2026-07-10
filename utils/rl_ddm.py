import numpy as np
import pandas as pd

def trust_drift(drift_intercept, drift_scaling, omega, expertise, q_domain, q_overall, alpha):
    """
    expertise: The baseline rating for the speaker in this statement's domain
    q_domain: The current Q-value for this statement's domain
    q_overall: The current overall trust Q-value
    """
    # NOTE: Does 'omega' parameter determine the general tendency to trust
    # domain-expertise over general trust?
    # TODO: do some simulations, see whether Q values are on different scales; add softmax if so

    domain_component = expertise * q_domain
    overall_component = q_overall
    
    combined_trust = (omega * domain_component) + ((1 - omega) * overall_component)
    
    return drift_intercept + drift_scaling * combined_trust

"""Simulate data using an RL-DDM that integrates the overall Q-value and the domain-specific Q-value.
Args:
    m_sim: model used for simulation
    n_trials: amount of trials to simulate
    n_sessions: amount of sessions to simulate
    reward_probs
    alpha: learning rate
    expertise_map: dict like {'Speaker1': {'A': 0.8, 'B': 0.2}, 'Speaker2': {'A': 0.3, 'B': 0.9}}
"""
def sim_rlddm_integrated(m_sim, ground_truth_df, alpha, expertise_map):
    """
    Runs a virtual participant through the already generated ground truth script.
    Args:
        m_sim: model used for simulation
        ground_truth_df: script
        alpha: learning rate
        expertise_map: dict like {'Speaker1': {'A': 0.8, 'B': 0.2}, 'Speaker2': {'A': 0.3, 'B': 0.9}}
    """
    # Initialize Q-values per speaker
    speakers = ground_truth_df['speaker'].unique()
    q_values = {
        spk: {'overall': 0.5, 'A': 0.5, 'B': 0.5} 
        for spk in speakers
    }

    results = []
    
    for i, row in ground_truth_df.iterrows():
        spk = row['speaker']
        dom = row['domain']
        is_true = row['is_true']
        exp_val = expertise_map[spk][dom]
        
        # 1. DDM Decision
        sol = m_sim.solve(conditions={
            "q_domain": q_values[spk][dom], 
            "q_overall": q_values[spk]['overall'], 
            "expertise": exp_val # changes drift rate for every trial
        })
        
        res = sol.sample(1).to_pandas_dataframe(drop_undecided=True)
        if res.empty: continue
        
        choice = res['choice'].iloc[0] # 1 = Trust, 0 = Distrust
        rt = res['RT'].iloc[0]
        print(f"DDM made choice: {choice}")
        # print(f"DDM params {sol.params}")
        # 2. ADDED: REWARD for alignment --> (Choice=Trust AND True) OR (Choice=Distrust AND False)
        #reward = 1 if choice == is_true else 0
        reward = 1 if is_true else 0
        
        # 3. RL Update
        q_values[spk]['overall'] = (1 - alpha) * q_values[spk]['overall'] + alpha * reward
        q_values[spk][dom] = (1 - alpha) * q_values[spk][dom] + alpha * reward
        
        # Combine ground truth with simulation results
        combined_row = row.to_dict()
        combined_row.update({
            "trial": i,
            "choice": choice,
            "rt": rt,
            "reward": reward,
            "expertise": exp_val,
            "domain": dom,
            "q_overall": q_values[spk]['overall'],
            "q_dom": q_values[spk][dom]
        })
        results.append(combined_row)
        
    return pd.DataFrame(results)
