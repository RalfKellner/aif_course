import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import matplotlib.pylab as plt
import gym
from gym import spaces
from sklearn.preprocessing import StandardScaler


# a class for downloading and processing of stock data
class DataLoader():
    def __init__(self, ticker, start_date, end_date, use_variables = [], trading_days = 252):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.use_variables = use_variables
        self.trading_days = trading_days
        
        self.load_data()
        self.preprocess_data()
        
        self.rnd_seeds = []

    def load_data(self):

        '''
        This function uses the yfinance data to download OHLC Volume data

        ticker, string: the abbreviation of companies at stock exchanges, e.g., AAPL
        start_date, string: the beginning of the time series, format is YYYY-MM-DD
        end_date, string: the end of the time series, format is YYYY-MM-DD
        '''

        # define the yfinance ticker instance
        tmp_ticker = yf.Ticker(self.ticker)
        # retrieve data
        tmp_data = tmp_ticker.history(start = self.start_date, end = self.end_date)
        # reduce data to OHLC-Volume data
        self.df = tmp_data[['Open', 'High', 'Low', 'Close', 'Volume']]
        self.raw_data = self.df.copy()

    def preprocess_data(self):

        '''
        This function determines daily returns and technical indicators. It is possible to use a subset of
        technical indicators by providing a list with names of these indicators. In the default setting
        indicators with a frequency of NAs which is higher than 5% are dropped

        use_variables, list of strings: names of technical indicators to use, e.g., ['AD', 'ABER_ZG_5_15']
        if None, all indicators are used
        '''

        # calculate returns
        self.df.loc[:, 'returns'] = np.log(self.df.Close) - np.log(self.df.Close.shift(1))

        # index of the return column to exclude raw data columns later
        idx = self.df.shape[1]

        # get all technical indicators first
        self.df.ta.strategy()

        # default case, determine all indicators and delete columns with too many missing values
        if len(self.use_variables) == 0:
            self.df = self.df.iloc[:, (idx - 1):]
            cols_to_drop = self.df.columns[self.df.isnull().mean() > 0.05]
            self.df.drop(cols_to_drop, axis = 1, inplace = True)
        # otherwise use variables as desired    
        else:
            self.df = self.df[['returns'] + self.use_variables]

        # monitor how many observations (rows) are lost by dropping rows with missing values
        n_obs_before = self.df.shape[0]
        self.df.dropna(inplace = True)
        n_obs_after = self.df.shape[0]
        dropped_obs = n_obs_before - n_obs_after

        print(f'{dropped_obs} observations have been deleted due to missing data')
        print(f'This equals {(dropped_obs / n_obs_before)*100:.2f}% of the data')

        # this ensures to have a decent amount of observations to train the agent
        assert len(self.df) > 3*self.trading_days, f'Data needs to inlude at least {3*self.trading_days} observations! Choose a longer time period...'
        self.min_values = self.df.min()
        self.max_values = self.df.max()

    
    def reset(self, seed = None):
        
        '''
        This function sets the step back to zero and randomly starts in the timeline of data.
        A seed can be given to reproduce results or if is not given is recorded at each reset
        '''
        high = len(self.df.index) - self.trading_days
        
        if seed:
            np.random.seed(seed)

        self.offset = np.random.randint(low = self.trading_days*2, high = high)
        self.scaler = StandardScaler()
        self.scaler.fit(self.df.drop(['returns'], axis = 1).iloc[(self.offset-self.trading_days*2):self.offset].values)
        self.step = 0
        
    def take_step(self):        
        """Returns data for current trading day and done signal"""
        state = self.df.drop(['returns'], axis = 1).iloc[self.offset + self.step].values.reshape(1, -1)
        state_scaled = self.scaler.transform(state).flatten()
        obs = np.concatenate((self.df[['returns']].iloc[self.offset + self.step].values, state_scaled), axis = 0)
        self.step += 1
        done = self.step > self.trading_days
        return obs, done
    
    # some utility functions, not for direct use
    def get_scaled_df_full_(self):
        self.full_scaler = StandardScaler()
        self.full_scaler.fit(self.df.drop(['returns'], axis = 1))
        states_scaled = self.full_scaler.transform(self.df.drop(['returns'], axis = 1))
        obs_full = np.concatenate((self.df[['returns']].values, states_scaled), axis = 1)
        return obs_full
    
    def financial_plot_(self):
        '''
        Function to visualize time series data, taken from here:
        https://towardsdatascience.com/basics-of-ohlc-charts-with-pythons-matplotlib-56d0e745a5be
        '''
        x = np.arange(0,len(self.raw_data))
        fig, (ax, ax2) = plt.subplots(2, figsize=(12,8), gridspec_kw={'height_ratios': [4, 1]})

        for idx, (dt, val) in enumerate(self.raw_data.iterrows()):
            color = '#2CA453'
            if val['Open'] > val['Close']: color= '#F04730'
            ax.plot([x[idx], x[idx]], [val['Low'], val['High']], color=color)
            ax.plot([x[idx], x[idx]-0.1], [val['Open'], val['Open']], color=color)
            ax.plot([x[idx], x[idx]+0.1], [val['Close'], val['Close']], color=color)

        # ticks top plot
        date_step = int(self.raw_data.shape[0] / 5)
        dates = [d.date() for d in self.raw_data.index]
        ax2.set_xticks(x[::date_step])
        ax2.set_xticklabels(dates[::date_step])
        ax.set_xticks(x, minor=True)# labels
        ax.set_ylabel('')
        ax2.set_ylabel('Volume')# grid
        ax.xaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)
        ax2.set_axisbelow(True)
        ax2.yaxis.grid(color='black', linestyle='dashed', which='both', alpha=0.1)# remove spines
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)# plot volume
        ax2.bar(x, self.raw_data['Volume'], color='lightgrey')
        # get max volume + 10%
        mx = self.raw_data['Volume'].max()*1.1
        # define tick locations - 0 to max in 4 steps
        yticks_ax2 = np.arange(0, mx+1, mx/4)
        # create labels for ticks. Replace 1.000.000 by 'mi'
        yticks_labels_ax2 = ['{:.2f} mi'.format(i/1000000) for i in yticks_ax2]
        ax2.yaxis.tick_right() # Move ticks to the left side
        # plot y ticks / skip first and last values (0 and max)
        plt.yticks(yticks_ax2[1:-1], yticks_labels_ax2[1:-1])
        plt.ylim(0,mx)

        # title
        ax.set_title(f'{self.ticker} stock price\n', loc='left', fontsize=20)
        # no spacing between the subplots
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.show()


# a class for the trading logic
class TradingSimulator:
    """ Implements core trading simulator for single-instrument univ """

    def __init__(self, trading_days, trading_cost_bps, time_cost_bps):
        # invariant for object life
        self.trading_cost_bps = trading_cost_bps
        self.time_cost_bps = time_cost_bps
        self.trading_days = trading_days

        # change every step
        self.step = 0
        self.actions = np.zeros(self.trading_days)
        self.navs = np.ones(self.trading_days)
        self.market_navs = np.ones(self.trading_days)
        self.strategy_returns = np.ones(self.trading_days)
        self.positions = np.zeros(self.trading_days)
        self.costs = np.zeros(self.trading_days)
        self.trades = np.zeros(self.trading_days)
        self.market_returns = np.zeros(self.trading_days)

    def reset(self):
        self.step = 0
        self.actions.fill(0)
        self.navs.fill(1)
        self.market_navs.fill(1)
        self.strategy_returns.fill(0)
        self.positions.fill(0)
        self.costs.fill(0)
        self.trades.fill(0)
        self.market_returns.fill(0)

    def take_step(self, action, market_return):
        """ Calculates NAVs, trading costs and reward
            based on an action and latest market return
            and returns the reward and a summary of the day's activity. """

        # this function gets the current position which is either -1, 0, 1 (short, cash, buy)
        # evaluates the wish for the next position which is either -1, 0, 1
        # it gets current navs, determines how many trades need to be executed to the 
        # desired position, going from cash to long or short is one trade, going from
        # short to long or long to short is two trades
        # the reward is based on the logic that at time t we get the information of the closing price
        # return and technical indicators, with that information we place our action and the 
        # corresponding position which is the starting position for the next day
        # implicitely we assume to buy the stock at the closing price at time t, a bold assumption
        # the reward is the return which is realized with the position from time t to t+1
        
        start_position = self.positions[max(0, self.step - 1)]
        start_nav = self.navs[max(0, self.step - 1)]
        start_market_nav = self.market_navs[max(0, self.step - 1)]
        self.market_returns[self.step] = market_return
        self.actions[self.step] = action

        end_position = action - 1  # short: 0, cash: 1, long: 2
        n_trades = end_position - start_position
        self.positions[self.step] = end_position
        self.trades[self.step] = n_trades

        # roughly value based since starting NAV = 1
        trade_costs = abs(n_trades) * self.trading_cost_bps
        time_cost = 0 if n_trades else self.time_cost_bps
        self.costs[self.step] = trade_costs + time_cost
        reward = start_position * market_return - self.costs[max(0, self.step-1)]
        self.strategy_returns[self.step] = reward

        if self.step != 0:
            self.navs[self.step] = start_nav * np.exp(self.strategy_returns[self.step])
            self.market_navs[self.step] = start_market_nav * np.exp(self.market_returns[self.step])

        info = {'reward': reward,
                'nav'   : self.navs[self.step],
                'costs' : self.costs[self.step]}

        self.step += 1
        return reward, info

    def result_(self):
        """returns current state as pd.DataFrame """
        return pd.DataFrame({'action'         : self.actions,  # current action
                             'nav'            : self.navs,  # starting Net Asset Value (NAV)
                             'market_nav'     : self.market_navs,
                             'market_return'  : self.market_returns,
                             'strategy_return': self.strategy_returns,
                             'position'       : self.positions,  # eod position
                             'cost'           : self.costs,  # eod costs
                             'trade'          : self.trades})  # eod trade)


# the full trading evironment class
class TradingEnvironment(gym.Env):
    """A simple trading environment for reinforcement learning.
    Provides daily observations for a stock price series
    An episode is defined as a sequence of 252 trading days with random start
    Each day is a 'step' that allows the agent to choose one of three actions:
    - 0: SHORT
    - 1: CASH
    - 2: LONG
    Trading has an optional cost (default: 10bps) of the change in position value.
    Going from short to long implies two trades.
    Not trading also incurs a default time cost of 1bps per step.
    An episode begins with a starting Net Asset Value (NAV) of 1 unit of cash.
    If the NAV drops to 0, the episode ends with a loss.
    If the NAV hits 2.0, the agent wins.
    The trading simulator tracks a buy-and-hold strategy as benchmark.
    
    The agent can choose among technical indicators as state variables.
    The pandas_ta package is used for calculating technical indicatos during data preprocessing.
    The default strategy is "all" which uses up to 276 technical indicators.
    Besides, the following default strategies are implemented: "candles", "cycles", "momentum", "overlap",
    "performance", "statistics", "trend", "volatility", "volume".
    
    """
    def __init__(self,
                 ticker='AAPL',
                 start_date = '2009-06-01',
                 end_date = '2015-12-31',
                 trading_days=252,
                 trading_cost_bps=1e-3,
                 time_cost_bps=1e-4,
                 use_variables = []):
        
        self.data_source = DataLoader(ticker = ticker,
                                      start_date = start_date,
                                      end_date = end_date,
                                      use_variables = use_variables,
                                      trading_days = trading_days)
        
        self.simulator = TradingSimulator(trading_days = trading_days,
                                          trading_cost_bps = trading_cost_bps,
                                          time_cost_bps = time_cost_bps)
        
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(self.data_source.min_values.values,
                                            self.data_source.max_values.values)
        self.reset()

    def step(self, action):
        """Returns state observation, reward, done and info"""
        assert self.action_space.contains(action), '{} {} invalid'.format(action, type(action))
        observation, done = self.data_source.take_step()
        reward, info = self.simulator.take_step(action=action,
                                                market_return=observation[0])
        return observation, reward, done, info

    def reset(self, seed = None):
        """Resets DataSource and TradingSimulator; returns first observation"""
        if seed:
            self.data_source.reset(seed = seed)
        else:
            self.data_source.reset()
        self.simulator.reset()
        return self.data_source.take_step()[0]



