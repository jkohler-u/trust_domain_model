import numpy as np
import pyddm

class LossRL(pyddm.LossFunction):
    name = "rl_loss"
    def setup(self, **kwargs):
         self.sessions = self.sample.condition_values('session')
         for s in self.sessions:
             trials = self.sample.subset(session=s).condition_values('trial')
             assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
         self.df =  self.sample.to_pandas_dataframe()
         self.sessdfs = [self.df.query(f'session == {s}').sort_values('trial') for s in self.sessions]
    def loss(self, model):
        likelihood = 0
        qleft = [.5]
        qright = [.5]
        alpha = model.get_dependence("drift").alpha
        for j in range(0, len(self.sessions)):
            sessdf = self.sessdfs[j]
            for i,row in sessdf.iterrows():
                chose_left = row['choice'] <= 0
                p = model.solve_analytical(conditions={"deltaq": qright[-1]-qleft[-1]}).evaluate(row['RT'], not chose_left)
                if chose_left:
                    qleft.append((1-alpha) * qleft[-1] + alpha * row['reward'])
                    qright.append(qright[-1])
                else: # Right choice
                    qright.append((1-alpha) * qright[-1] + alpha * row['reward'])
                    qleft.append(qleft[-1])
                if p <= 0:
                    return -np.inf
                likelihood += np.log(p)
        model.last_qleft = qleft
        model.last_qright = qright
        return -likelihood
    
class LossRLFast(pyddm.LossFunction):
    """
    A faster version of the RL loss function that rounds the number of decimal digits of the detla_Q-value to two
    """
    name = "rl_loss"
    ROUND = 2 # Number of decimal digits to round to, lower number gives better performance but lower accuracy
    def setup(self, **kwargs):
         self.sessions = self.sample.condition_values('session')
         for s in self.sessions:
             trials = self.sample.subset(session=s).condition_values('trial')
             assert set(trials) == set(range(min(trials), min(trials)+len(trials))), "Trials must be sequentially numbered"
         self.df =  self.sample.to_pandas_dataframe().sort_values(["session", "trial"]).reset_index()
         self.df['deltaq'] = 0
         self.sessinds = [self.df['session'] == s for s in self.sessions]
    def loss(self, model):
        likelihood = 0
        alpha = model.get_dependence("drift").alpha
        for j in range(0, len(self.sessions)):
            qleft = [.5]
            qright = [.5]
            for _,row in self.df[self.sessinds[j]].iterrows():
                chose_left = row['choice'] <= 0
                if chose_left:
                    qleft.append((1-alpha) * qleft[-1] + alpha * row['reward'])
                    qright.append(qright[-1])
                else: # Right choice
                    qright.append((1-alpha) * qright[-1] + alpha * row['reward'])
                    qleft.append(qleft[-1])
            self.df.loc[self.sessinds[j],'deltaq'] = np.round(np.asarray(qright)[:-1] - np.asarray(qleft)[:-1], self.ROUND)
        likelihood = pyddm.get_model_loss(model=model, sample=pyddm.Sample.from_pandas_dataframe(self.df, 'RT', 'choice'), lossfunction=pyddm.LossLikelihood)
        model.last_qleft = np.asarray(qleft)[:-1]
        model.last_qright = np.asarray(qright)[:-1]
        return likelihood