import pandas as pd
import numpy as np

# ADDED
def generate_ground_truth(n_sessions, n_trials_per_speaker, speakers, domains, truth_probs):
    """
    Creates the 'script' of the experiment.
    Args:
        n_sessions: amount of sessions to generate
        n_trials_per_speaker: amount of trials per speaker
        speakers: List containing the speakers
        domains: List containing the domains
        truth_probs: Dictionary containing probability that statements are true/false {'Speaker1': {'A': 0.8, 'B': 0.2}, ...}
    """
    all_trials = []
    
    for sess in range(n_sessions):
        for speaker in speakers: # Speakers happen one after the other
            for t in range(n_trials_per_speaker):
                domain = np.random.choice(domains) # Domains are random
                
                # Determine if the tweet is actually TRUE or FALSE
                prob_true = truth_probs[speaker][domain]
                is_true = int(np.random.uniform() < prob_true)
                
                all_trials.append({
                    "session": sess,
                    "speaker": speaker,
                    "domain": domain,
                    "is_true": is_true
                })
    
    return pd.DataFrame(all_trials)