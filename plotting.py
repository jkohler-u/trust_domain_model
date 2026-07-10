import numpy as np
import matplotlib.pyplot as plt

"""
input values: 
DDM: 
t_0: 0.3 - no need to vary, similar across all participants
z/bias: perceived_domain_expertise_speaker of the speaker
a/caution: general confidence of the participant
drift rate: predicted by the RL → Q

RL: 
learning rate
omega - how we value domain vs general expertise
drift_rate_intercept - baseline trust
drift_rate_scaling - sensitivity to the recommendation of the RF angent

output: 
RT
choice
"""


"""
Simulate data for: 
objectively trustworthy agent: 92/100 correct 
objectivly untrastowrthy agent: 4/100 correct 
plot: 
RT and choice over time - cummulative accuracy
=> for both we would expect the RT to decrease over the trials and the accuracy to increase
=> Given our setup this effect should be slightly stronger for the objectively trustworthy agent
 
"""

tweets = np.random.choice([1, -1], size=100)
choices = np.random.choice([1, -1], size=100)
rts = np.random.uniform(0.3, 3.0, 100)

def plot_cummulative_accuracy(tweets, choices): 
    accuracy = (choices == tweets).astype(int)
    cumulative_acc = np.cumsum(accuracy) / np.arange(1, len(accuracy) + 1)

    plt.figure(figsize=(10, 4))
    plt.plot(cumulative_acc, color='green', linewidth=2)
    plt.axhline(y=0.5, color='red', linestyle='--', label='Chance Level')
    plt.xlabel('Trial Number')
    plt.ylabel('Cumulative Accuracy')
    plt.title('Learning Curve (Cumulative Accuracy)')
    plt.legend()
    plt.ylim(0, 1)
    plt.show()
# plot_cummulative_accuracy(tweets, choices)

def plot_cummulative_accuracy_rts(tweets, choices, rts): 
    # 1. Calculate Cumulative Accuracy
    accuracy = (choices == tweets).astype(int)
    cumulative_acc = np.cumsum(accuracy) / np.arange(1, len(accuracy) + 1)

    # 2. Calculate Cumulative Mean RT (to smooth out the noise)
    cumulative_rt = np.cumsum(rts) / np.arange(1, len(rts) + 1)

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Axis 1: Cumulative Accuracy (Left Y-axis)
    color_acc = 'tab:blue'
    ax1.set_xlabel('Trial Number')
    ax1.set_ylabel('Cumulative Accuracy', color=color_acc, fontsize=12)
    ax1.plot(cumulative_acc, color=color_acc, linewidth=3, label='Accuracy')
    ax1.tick_params(axis='y', labelcolor=color_acc)
    ax1.set_ylim(0, 1)
    ax1.axhline(y=0.5, color='grey', linestyle='--', alpha=0.5, label='Chance')

    # Create a second axis that shares the same x-axis
    ax2 = ax1.twinx() 

    # Axis 2: Cumulative RT (Right Y-axis)
    color_rt = 'tab:red'
    ax2.set_ylabel('Cumulative Mean RT (s)', color=color_rt, fontsize=12)
    ax2.plot(cumulative_rt, color=color_rt, linewidth=3, label='Reaction Time')
    ax2.tick_params(axis='y', labelcolor=color_rt)

    # Formatting
    plt.title('Learning Progress: Accuracy vs. Reaction Time', fontsize=14)
    fig.tight_layout() 
    plt.show()
# plot_cummulative_accuracy_rts(tweets, choices, rts)


"""
Simulate data for: 
objectively trustworthy agent within domain, av. trustworthyness outside domain:
 - 48/50 correct within domain of expertise
 - 25/50 correct outside domain of expertise

objectively trustworthy agent outside domain, av. trustworthyness within domain:
 - 25/50 correct within domain of expertise
 - 48/50 correct outside domain of expertise
plot: 
RT and choice over time 
"""
tweet_domain = np.array([1]*50 + [2]*50)
np.random.shuffle(tweet_domain)
def plot_cumulative_accuracy_per_domain(tweet, domain, choice):
    """
    tweet - whether the tweet was true/false
    domain - what domain the tweet belongs to
    choice - the choice the user made 
    """
    # Filter data per domain
    acc_dom1 = (tweet[domain == 'A'] == choice[domain ==  'A']).astype(int)
    acc_dom2 = (tweet[domain == 'B'] == choice[domain ==  'B']).astype(int)
    print(acc_dom1[:10])
    # Calculate cumulative means
    cum_acc1 = np.cumsum(acc_dom1) / np.arange(1, len(acc_dom1) + 1)
    cum_acc2 = np.cumsum(acc_dom2) / np.arange(1, len(acc_dom2) + 1)

    # Create subplots: 1 row, 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # Plot Domain 1
    ax1.plot(cum_acc1, color='blue', linewidth=2)
    ax1.axhline(y=0.5, color='red', linestyle='--', label='Chance')
    ax1.set_title('Domain 1 Learning Curve')
    ax1.set_xlabel('Trial Number (in Domain)')
    ax1.set_ylabel('Cumulative Accuracy')
    ax1.set_ylim(0, 1)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot Domain 2
    ax2.plot(cum_acc2, color='green', linewidth=2)
    ax2.axhline(y=0.5, color='red', linestyle='--', label='Chance')
    ax2.set_title('Domain 2 Learning Curve')
    ax2.set_xlabel('Trial Number (in Domain)')
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# plot_cumulative_accuracy_per_domain(tweets, tweet_domain, choices)
def plot_trust_switching(q_values, choices, tweets, domain_labels=None):
    trials = np.arange(1, len(q_values) + 1)
    q_values = np.array(q_values)
    
    plt.figure(figsize=(14, 6))
    
    # 1. Plot the Q-value line
    plt.plot(trials, q_values, color='black', lw=2, alpha=0.7, label='Learned Trust (Q)', zorder=1)
    
    # 2. Plot the decisions as markers ON the line
    trust_idx = np.where(np.array(choices) == 1)[0]
    distrust_idx = np.where(np.array(choices) == 0)[0]

    plt.scatter(trust_idx + 1, q_values[trust_idx], 
                color='forestgreen', marker='o', s=40, label='Decision: Trust', zorder=2)
    plt.scatter(distrust_idx + 1, q_values[distrust_idx], 
                color='crimson', marker='x', s=40, label='Decision: Distrust', zorder=2)

    # --- FIX: Line of markers at the bottom representing the ground truth (tweets) ---
    corr_ix = np.where(np.array(tweets) == 1)[0]
    false_ix = np.where(np.array(tweets) == 0)[0]
    
    # Calculate a y-position that is slightly below the lowest Q-value
    bottom_y = np.min(q_values) - (np.ptp(q_values) * 0.1) 
    
    plt.scatter(corr_ix + 1, [bottom_y] * len(corr_ix), 
                color='forestgreen', marker='o', s=20, alpha=0.5, label='Truth: Correct', zorder=2)
    plt.scatter(false_ix + 1, [bottom_y] * len(false_ix), 
                color='crimson', marker='x', s=20, alpha=0.5, label='Truth: False', zorder=2)

    # 3. Add a reference line at 0
    plt.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # 4. Formatting
    plt.title("The Trust Switching Point: Internal Value vs. Final Decision", fontsize=14)
    plt.xlabel("Trial Number", fontsize=12)
    plt.ylabel("Q-Value (Trustworthiness Estimate)", fontsize=12)
    
    # Clean up legend to avoid duplicates
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='best', frameon=False)
    
    plt.grid(True, alpha=0.2)
    plt.show()

qvalue_overall = np.random.uniform(-5, 5, 100)
# plot_trust_switching(qvalue_overall, choices)



""" ---- Form the course ---- """
def _pretty_ticks(n, max_ticks=10):
    if n <= 1: return [1]
    idx = np.linspace(0, n - 1, max_ticks)
    return np.unique(np.clip(np.round(idx) + 1, 1, n).astype(int)).tolist()

def plot_qvalues(qvalue_overall, qvalue_option1, qvalue_option2, show_legend=False, option_names=None):
    """ Plots the development of the qvalues over time using subplots to handle different lengths """
    
    # Create a figure with 2 subplots (1 row, 2 columns)
    # ax1 for the global value, ax2 for the specific option values
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # --- Plot 1: Overall Q-Value ---
    n_overall = len(qvalue_overall)
    x_overall = np.arange(1, n_overall + 1)
    ax1.plot(x_overall, qvalue_overall, color='black', lw=2)
    ax1.set_title("Overall Learned Value")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Estimated Value")
    ax1.set_xlim(1, n_overall)
    ax1.set_xticks(_pretty_ticks(n_overall))
    ax1.grid(True, alpha=0.3)

    # --- Plot 2: Option Specific Q-Values ---
    # We put both option 1 and option 2 on this plot
    # Note: Since they might have different lengths, we plot them separately
    n_opt1 = len(qvalue_option1)
    n_opt2 = len(qvalue_option2)
    
    ax2.plot(np.arange(1, n_opt1 + 1), qvalue_option1, color='blue', lw=2, label=option_names[1] if option_names else "Domain A")
    ax2.plot(np.arange(1, n_opt2 + 1), qvalue_option2, color='red', lw=2, label=option_names[2] if option_names else "Domain B")
    
    ax2.set_title("Option-Specific Learned Values")
    ax2.set_xlabel("Trial")
    ax2.set_ylabel("Estimated Value")
    
    # X-axis limit is based on the longest option array
    max_opt_len = max(n_opt1, n_opt2)
    ax2.set_xlim(1, max_opt_len)
    ax2.set_xticks(_pretty_ticks(max_opt_len))
    ax2.grid(True, alpha=0.3)

    if show_legend:
        ax2.legend(loc="best", frameon=False)

    plt.tight_layout()
    plt.show()

qvalue_option1 = np.random.uniform(-5, 5, 50)
qvalue_option2 = np.random.uniform(-5, 5, 50)
# plot_qvalues(qvalue_overall, qvalue_option1 , qvalue_option2, show_legend=True, option_names=["Overall", "Opt 1", "Opt 2"])



def plot_pred_errors(q_0, q_d1, q_d2, domain, tweets, ax=None):
    # 1. Unpack the Q-values
    # qvalues is assumed to be a matrix/list: [Q_overall, Q_dom1, Q_dom2]
    
    # 2. Calculate Prediction Errors
    # Note: we must ensure tweets are sliced the same way as qvalues
    pred_err_q_0 = tweets - q_0
    pred_err_q_d1 = tweets[domain == 'A'] - q_d1
    print(pred_err_q_0[:10])
    pred_err_q_d2 = tweets[domain == 'B'] - q_d2
    
    # 3. Create Subplots (1 row, 3 columns)
    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Data to iterate through for the plots
    errors = [pred_err_q_0, pred_err_q_d1, pred_err_q_d2]
    titles = ["Overall PE", "Domain 1 PE", "Domain 2 PE"]
    colors = ['black', 'blue', 'red']
    
    for i in range(3):
        data = errors[i]
        n_trials = len(data)
        x = np.arange(1, n_trials + 1)
        
        axs[i].plot(x, data, color=colors[i], lw=2)
        axs[i].axhline(0, color='black', linestyle='--', alpha=0.5) # Zero line
        axs[i].set_title(titles[i])
        axs[i].set_xlabel("Trial (in context)")
        axs[i].set_ylabel("Prediction Error")
        axs[i].set_xlim(1, n_trials)
        axs[i].set_xticks(_pretty_ticks(n_trials))
        axs[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()
# plot_pred_errors(qvalue_overall, qvalue_option1, qvalue_option2, tweet_domain, tweets)


def plot_outcomes(choices, tweets):
    """
    choices: array of 1 (Trust) and -1 (Distrust)
    tweets: array of 1 (Trust was correct) and -1 (Distrust was correct)

    If a blue dot is at +1, the user trusted and the tweet was actually true. (Success)
    If a blue dot is at -1, the user trusted but the tweet was a lie. (Failure)

    The Red X's (Distrust):

    If a red X is at +1, the user distrusted and the tweet was actually a lie. (Success)
    If a red X is at -1, the user distrusted but the tweet was actually true. (Failure)


    """
    # Convert to numpy arrays for easy indexing
    choices = np.asarray(choices)
    tweets = np.asarray(tweets)
    n_trials = len(choices)
    x = np.arange(1, n_trials + 1)

    # Calculate the actual outcome received for each trial
    # If choice matches tweet (1==1 or -1==-1), outcome is 1 (Correct)
    # If they differ, outcome is -1 (Incorrect)
    outcomes = np.where(choices == tweets, 1, -1)

    plt.figure(figsize=(12, 5))
    
    # 1. Plot the "Truth" (The Grey Lines)
    # Since we have two options (Trust/Distrust), 
    # the reward for Trusting is 'tweets' and the reward for Distrusting is '-tweets'
    plt.plot(x, tweets, color='gray', lw=1, alpha=0.5, label='Reward if Trust')
    plt.plot(x, -tweets, color='gray', lw=1, alpha=0.3, ls='--', label='Reward if Distrust')

    # 2. Plot the actual choices as dots
    # We use different colors for Trust and Distrust
    trust_mask = (choices == 1)
    distrust_mask = (choices == -1)

    # Dots for Trusting
    plt.scatter(x[trust_mask], outcomes[trust_mask], 
                color='blue', marker='o', s=50, label='Chose Trust', zorder=3)
    
    # Dots for Distrusting
    plt.scatter(x[distrust_mask], outcomes[distrust_mask], 
                color='red', marker='x', s=50, label='Chose Distrust', zorder=3)
    plt.show()
    
# plot_outcomes(choices, tweets)



