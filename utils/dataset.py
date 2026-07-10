import pandas as pd
import numpy as np

# ADDEDimport numpy as np
import pandas as pd
import random

def generate_ground_truth(n_sessions, n_trials_per_speaker, speakers, domains, truth_probs):
    """
    Creates the 'script' of the experiment.
    Args:
        n_sessions: amount of sessions to generate
        n_trials_per_speaker: amount of trials per speaker (should be even)
        speakers: List containing the speakers
        domains: List containing the domains (assumed to be 2 domains)
        truth_probs: Dictionary containing probability that statements are true/false
    """
    # Safety check: if we want exactly half, n_trials_per_speaker must be even
    if n_trials_per_speaker % 2 != 0:
        raise ValueError("n_trials_per_speaker must be an even number to split domains exactly 50/50.")

    all_trials = []
    
    # Calculate how many of each domain we need per speaker
    half = n_trials_per_speaker // 2
    
    for sess in range(n_sessions):
        for speaker in speakers:
            # 1. Create a fixed list: half of domain A, half of domain B
            # This works for any number of domains if you divide by len(domains)
            trial_domains = []
            for dom in domains:
                trial_domains.extend([dom] * half)
            
            # 2. Shuffle the list so the order is random, but the count is exact
            random.shuffle(trial_domains)
            
            # 3. Assign truth values based on the pre-determined domain list
            for domain in trial_domains:
                prob_true = truth_probs[speaker][domain]
                is_true = int(np.random.uniform() < prob_true)
                
                all_trials.append({
                    "session": sess,
                    "speaker": speaker,
                    "domain": domain,
                    "is_true": is_true
                })
    
    return pd.DataFrame(all_trials)