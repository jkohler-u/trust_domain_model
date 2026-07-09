import numpy as np
import pandas as pd
import pyddm




def sim_rlddm(m_sim, n_trials, n_sessions, reward_probabilities, alpha):
    qvals = np.array([0.5, 0.5])
    choice = []
    reward = []
    trial = []
    session = []
    rt = []
    qleft = []
    qright = []
    for sess in range(0, n_sessions):
        for t in range(0, n_trials):
            # compute choice probabilities using softmax formula
            sol = m_sim.solve(conditions={"deltaq": qvals[1]-qvals[0]})
            res = sol.sample(5).to_pandas_dataframe(drop_undecided=True) # We only use the first non-undecided trial
            r = np.random.randint(0, 5)
            choice.append(res['choice'][r]) # 1 for right and 0 for left
            rt.append(float(res['RT'][r]))
            session.append(sess)
            trial.append(t)
            reward.append(int(np.random.uniform() < reward_probabilities[choice[-1]]))
            qvals[choice[-1]] = (1-alpha)*qvals[choice[-1]] + alpha*reward[-1]
            qleft.append(qvals[0])
            qright.append(qvals[1])
    df = pd.DataFrame({"trial": trial, "session": session, "choice": choice, "reward": reward, "rt": rt})
    return pyddm.Sample.from_pandas_dataframe(df, choice_column_name="choice", rt_column_name="rt")



