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

def sim_rlddm_integrated(m_sim, n_trials, n_sessions, reward_probs, alpha, expertise_map):
    """
    Simulate data using an RL-DDM that integrates the overall Q-value and the domain-specific Q-value.
     Args:
        m_sim: model used for simulation
        n_trials: amount of trials to simulate
        n_sessions: amount of sessions to simulate
        reward_probs
        alpha: learning rate
        expertise_map: dict like {'Speaker1': {'A': 0.8, 'B': 0.2}, 'Speaker2': {'A': 0.3, 'B': 0.9}}
    """
    speakers = list(expertise_map.keys())
    
    # Initialize Q-values: One overall and one per domain per speaker
    q_values = {
        spk: {'overall': 0.5, 'A': 0.5, 'B': 0.5} 
        for spk in speakers
    }
    
    data = []
    
    for sess in range(n_sessions):
        for t in range(n_trials):
            # 1. Randomly assign Speaker AND Domain for this trial
            speaker = np.random.choice(speakers)
            domain = np.random.choice(['A', 'B'])
            
            # Get the specific expertise for this speaker in this domain
            exp_val = expertise_map[speaker][domain]
            
            # 2. Get the current Q-values for THIS speaker
            q_dom = q_values[speaker][domain]
            q_ovr = q_values[speaker]['overall']
            
            # Pass current state to the model
            sol = m_sim.solve(conditions={
                "q_domain": q_dom, 
                "q_overall": q_ovr, 
                "expertise": exp_val
            })
            
            res = sol.sample(1).to_pandas_dataframe(drop_undecided=True)
            if res.empty: continue 
            
            choice = res['choice'].iloc[0]
            rt = res['RT'].iloc[0]
            
            # Reward based on ground truth
            reward = int(np.random.uniform() < reward_probs[choice])
            
            # 3. UPDATE (Only for the speaker who just spoke, only for domain that just domained)
            q_values[speaker]['overall'] = (1 - alpha) * q_ovr + alpha * reward
            q_values[speaker][domain] = (1 - alpha) * q_dom + alpha * reward
            
            data.append({
                "trial": t, 
                "session": sess, 
                "choice": choice, 
                "reward": reward, 
                "rt": rt, 
                "domain": domain, 
                "speaker": speaker,
                "expertise": exp_val
            })
            
    return pd.DataFrame(data)

# def sim_rlddm_integrated(m_sim, n_trials, n_sessions, reward_probs, alpha, expertise_map):
#     """
#     Simulate data using an RL-DDM that integrates the overall Q-value and the domain-specific Q-value.
#     Args:
#         - m_sim: model used for simulation
#         - n_trials: amount of trials to simulate
#         - n_sessions: amount of sessions to simulate
#         - reward_probs
#         - alpha: learning rate
#         - expertise_map:
#     """
#     q_overall = 0.5
#     q_domains = {'A': 0.5, 'B': 0.5}
    
#     data = []
    
#     for sess in range(n_sessions):
#         for t in range(n_trials):
#             # Randomly assign domain for this trial
#             domain = np.random.choice(['A', 'B'])
#             exp_val = expertise_map[domain]
            
#             # Pass current state to the model
#             sol = m_sim.solve(conditions={
#                 "q_domain": q_domains[domain], 
#                 "q_overall": q_overall, 
#                 "expertise": exp_val
#             })
            
#             # Sample a response
#             res = sol.sample(1).to_pandas_dataframe(drop_undecided=True)
#             if res.empty: continue # Handle undecided
            
#             choice = res['choice'].iloc[0]
#             rt = res['RT'].iloc[0]
            
#             # Reward based on the ground truth of the statement
#             reward = int(np.random.uniform() < reward_probs[choice])
            
#             # --- UPDATE RULES ---
#             q_overall = (1 - alpha) * q_overall + alpha * reward
#             q_domains[domain] = (1 - alpha) * q_domains[domain] + alpha * reward
            
#             data.append({
#                 "trial": t, "session": sess, "choice": choice, 
#                 "reward": reward, "rt": rt, "domain": domain, "expertise": exp_val
#             })
            
#     return pd.DataFrame(data)