import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import matplotlib.lines as mlines
import tensorflow as tf

# a function for eavluation of agent's trading performance
def ai_trade_performance(environment, agent, seeds = [], num_plays = 20, plot_results = True):

    '''
    This function uses a given environment and a trained agent to examine and compare the agent's performance 
    for num_plays episodes. Every episode, the agent makes trading decisions for the number of trading days
    which is specified withing the environment (usually 252 days). Assuming a starting value of 1 monetary unit,
    the final net asset value is the multiple of 1 monetary unit which is generated by means of trading decisions.
    The function outputs a comparison of the agent's performance and the buy and hold strategy as well as two lists
    containing the final net asset values of all episodes played

    Input:
    environment: a predefined stock environment from the aif_course package
    agetn: a (trained) model from the stable baselines 3 package
    seeds: provide a list of integer numbers with equal length to num_plays, if you want to compare the same episodes
    num_plays: the number of episodes which are played and whose performance are evaluated
    plot_results: set to false if you are not interested in the visual output

    Output:
    results: a pandas table which shows the agent's average of final net asset values, the avg. buy and hold final net asset values
             the fraction of episodes for which the agent created a higher final net asset value;   
             the second row compares the same thing, but, it scales the average performance with the standard deviation of final net asset values
    f_navs: a list with final net asset values of the agent for each episode
    m_navs: a list with final net asset values of the buy and hold strategy for each episode
    '''

    final_navs = []
    final_market_navs = []

    if len(seeds) != 0:
        assert len(seeds) == num_plays, 'Number of plays needs to be the same as the length of the random seed list'
    else: 
        seeds = np.random.randint(low = 0, high = 1000, size = num_plays)

    for i in range(num_plays):
        s = environment.reset(seed = seeds[i])
        done = False

        while not(done):
            a, pred = agent.predict(s, deterministic = True)
            s, r, done, info = environment.step(a)

        navs_tmp = environment.simulator.navs.copy()
        market_navs_tmp = environment.simulator.market_navs.copy()
        final_navs.append(navs_tmp)
        final_market_navs.append(market_navs_tmp)

    f_navs = np.array([f_nav[-1] for f_nav in final_navs])
    m_navs = np.array([m_nav[-1] for m_nav in final_market_navs])
    
    f_retrisk = [(np.mean(final_navs) - 1) / np.std(final_navs, ddof = 1)]
    m_retrisk = [(np.mean(final_market_navs) - 1) / np.std(final_market_navs, ddof = 1)]
    
    agent_mean = np.mean([f_nav[-1] for f_nav in final_navs])
    market_mean = np.mean([m_nav[-1] for m_nav in final_market_navs])
    agent_better = np.mean(f_navs > m_navs)
    
    agent_retrisk = np.mean(f_retrisk)
    market_retrisk = np.mean(m_retrisk)
    agent_retrisk_better = np.mean(f_retrisk > m_retrisk)
    
    y_lim_max = max(max(f_navs), max(m_navs))

    if plot_results:
        fig, axs = plt.subplots(1, 3, figsize = (18, 6))

        for i in range(num_plays):
            axs[0].plot(final_navs[i])
            axs[0].set_title('Trained agent')
            axs[0].set_xlabel('Trading days')
            axs[0].set_ylabel('Net asset value')
            axs[0].set_ylim(0, y_lim_max)
            axs[1].plot(final_market_navs[i])
            axs[1].set_title('Buy and hold')
            axs[1].set_xlabel('Trading days')
            axs[1].set_ylabel('Net asset value')
            axs[1].set_ylim(0, y_lim_max)
            axs[2].scatter(final_market_navs[i][-1], final_navs[i][-1])
            axs[2].set_title('Buy and hold vs. Agent')
            axs[2].set_xlabel('Buy and hold NAV')
            axs[2].set_ylabel('Agent NAV')
            axs[2].set_xlim(0, y_lim_max)
            axs[2].set_ylim(0, y_lim_max)
            line = mlines.Line2D([0, 1], [0, 1], color='grey')
            transform = axs[2].transAxes
            line.set_transform(transform)
            axs[2].add_line(line)
        plt.show()
        
    results_data = [
        [agent_mean, market_mean, agent_better],
        [agent_retrisk, market_retrisk, agent_retrisk_better ]
    ]
    
    results = pd.DataFrame(data = results_data, columns = ['agent_avg', 'market_avg', 'agent_better'])
    results.index = ['return', 'return_risk']
    
    return results, f_navs, m_navs

# a function to analyze actions taken by a trained agent
def analyze_actions_taken(environment, model, plot_actions = False, environment_new = False):

    '''
    Use this function to evaluate how many times the trained agent would choose the actions 
    short, cash and long over the whole time frame of the specified environment
    '''
    
    # get all scaled state variable realizations
    states = environment.data_source.get_scaled_df_full_()
    if environment_new == True:
        # get a list of all actions taken in each state
        actions_taken, _ = model.predict(states[:, 1:], deterministic = True)
    else:
        # get a list of all actions taken in each state
        actions_taken, _ = model.predict(states, deterministic = True)
    
    # generate a dictionary with position: count keys and values
    labels = {0: 'short', 1:'cash', 2: 'long'}
    x, counts = np.unique(actions_taken, return_counts=True)
    labels_to_use = [labels[i] for i in x]
    action_counts = {}
    for i, j in zip(x, counts):
        action_counts[labels[i]] = j
    
    # if you set the plot_actions variable to True, you get a barchart of actions taken
    if plot_actions:
        plt.bar(labels_to_use, counts, color = ['purple', 'goldenrod', 'olivedrab'])
        plt.title('chosen actions over time')
        plt.xlabel('action')
        plt.ylabel('count')
        plt.show()

    return action_counts


# a helper function for getting the gradients of the a2c network
def get_action_gradients_(states_tf, action_vec, actor_net):
    with tf.GradientTape() as tape:
        tape.watch(states_tf)
        action_probs = actor_net(states_tf)
        actions = tf.one_hot(action_vec, depth = 3)
        out = tf.reduce_sum(tf.multiply(action_probs, actions), axis = 1)

    gradients = tape.gradient(out, states_tf)
    return gradients

# a helper function to get normalized feature importance    
def feature_importance_(gradients_np):
    non_normal_importance = np.sqrt(np.mean(gradients_np**2, axis = 0)) 
    sign_importance = np.sign(np.mean(gradients_np, axis = 0))
    normal_importance = (non_normal_importance / np.sum(non_normal_importance)) * sign_importance
    return normal_importance

# a function for getting feature importances of all actions for from the actor network of the a2c model
def feat_importance_a2c(model, environment):
    params =  [
        model.get_parameters()['policy']['mlp_extractor.policy_net.0.weight'].cpu().detach().numpy().transpose(),
        model.get_parameters()['policy']['mlp_extractor.policy_net.0.bias'].cpu().detach().numpy(),
        model.get_parameters()['policy']['mlp_extractor.policy_net.2.weight'].cpu().detach().numpy().transpose(),
        model.get_parameters()['policy']['mlp_extractor.policy_net.2.bias'].cpu().detach().numpy(),
        model.get_parameters()['policy']['action_net.weight'].cpu().detach().numpy().transpose(),
        model.get_parameters()['policy']['action_net.bias'].cpu().detach().numpy()
    ]

    actor_net = tf.keras.Sequential([
        tf.keras.layers.Dense(params[0].shape[1], activation = 'tanh', input_shape = (environment.data_source.df.shape[1],)),
        tf.keras.layers.Dense(params[2].shape[1], activation = 'tanh'),
        tf.keras.layers.Dense(params[4].shape[1], activation = 'softmax')
    ])
    
    actor_net.set_weights(params)
    
    states = environment.data_source.get_scaled_df_full_()
    states_tf = tf.Variable(states)
    
    actions_taken, _ = model.predict(states, deterministic = True)
        
    fis = []
    for i in range(3):
        actions = [i] * environment.data_source.df.shape[0]
        grads = get_action_gradients_(states_tf, actions, actor_net).numpy()
        fis.append(feature_importance_(grads))

    fis.append(feature_importance_(get_action_gradients_(states_tf, actions_taken, actor_net).numpy()))
    
    all_importances = pd.DataFrame(fis).transpose()
    all_importances.columns = ['short', 'cash', 'long', 'taken']
    all_importances.index = environment.data_source.df.columns
    
    return all_importances

    
