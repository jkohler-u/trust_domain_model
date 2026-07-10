import numpy as np
import pyddm

class LossRLIntegrated(pyddm.LossFunction):
    name = "rl_integrated_loss"

    def setup(self, **kwargs):
        self.df = self.sample.to_pandas_dataframe().sort_values(["session", "trial"])
        self.sessions = self.df['session'].unique()

    def loss(self, model):
        likelihood = 0
        # Get parameters from the model
        params = model.parameters() 
        drift_params = params.get('drift', {})
        alpha = drift_params.get('alpha') 
        
        for sess in self.sessions:
            sessdf = self.df[self.df['session'] == sess]
            
            # TODO: Do we want to reset Q-values at start of session 
            # or keep across sessions?
            q_values = {
                'Speaker1': {'overall': 0, 'A': 0, 'B': 0}, 
                'Speaker2': {'overall': 0, 'A': 0, 'B': 0}
            }
            
            for _, row in sessdf.iterrows():
                current_spk = row['speaker']
                dom = row['domain']
                exp = row['expertise']
                
                # Current conditions for this specific trial
                conditions = {
                    "q_domain": q_values[current_spk][dom], 
                    "q_overall": q_values[current_spk]['overall'], 
                    "expertise": exp
                }
                
                # Calculate probability of the observed RT and Choice
                # NOTE: choice == 1 is "Yes"/"Trust"
                p = model.solve_analytical(conditions=conditions).evaluate(row['RT'], row['choice'] == 1)
                
                # RL Updates
                reward = row['reward']
                q_values[current_spk]['overall'] = (1 - alpha) * q_values[current_spk]['overall'] + alpha * reward
                q_values[current_spk][dom] = (1 - alpha) * q_values[current_spk][dom] + alpha * reward
                
                if p <= 0: return -np.inf
                likelihood += np.log(p)
                
        return -likelihood