from backtesting import Backtest, Strategy
from matplotlib import pyplot as plt
import numpy as np
import mplfinance as mpf

class strategy(Strategy):
    """ STRATEGY PARAMETERS """
    n_candles = 10  # Number of consecutive increases to trigger a buy
    q_candles = 3   # Number of consecutive increases while a buy position is active to trigger a sell
    x_candles = 5   # Number of consecutive decreases while a buy position is active to trigger a sell
    min_sharpness = 0  # Minimum sharpness for a valid increase
    loss_threshold = 0.1  # Maximum allowable loss before selling

    def init(self):
        """ Initialize Strategy Variables """
        self.index = -1  # Tracks the index of the current candle being processed
        self.increase_streak = 0  # Tracks consecutive increases
        self.decrease_streak = 0  # Tracks consecutive decreases
        self.entry_price = None  # Tracks the price at which the position was entered
        self.trade_history = []  # Tracks buy and sell trades

    def next(self):
        """ Process Each Candle """
        self.index += 1

        # Skip if there is no previous candle
        if self.index == 0:
            return

        # Get the current and previous close prices
        current_close = self.data.Close[self.index]
        previous_close = self.data.Close[self.index - 1]

        # Calculate sharpness
        sharpness = abs(current_close - previous_close) / previous_close

        # Update streaks
        self.update_streaks(current_close, previous_close, sharpness)

        # Check for buy condition
        self.check_buy_condition(current_close)

        # Check for sell conditions
        self.check_sell_conditions(current_close)

    def update_streaks(self, current_close, previous_close, sharpness):
        """ Update Increase and Decrease Streaks """
        if current_close > previous_close and sharpness >= self.min_sharpness:
            self.increase_streak += 1
            self.decrease_streak = 0  # Reset decrease streak
        elif current_close < previous_close:
            self.decrease_streak += 1
            self.increase_streak = 0  # Reset increase streak
        else:
            # Reset both streaks if the price is unchanged
            self.increase_streak = 0
            self.decrease_streak = 0

    def check_buy_condition(self, current_close):
        """ Check and Execute Buy Condition """
        if not self.position and self.increase_streak >= self.n_candles:
            self.buy()
            self.entry_price = current_close
            self.trade_history.append({'index': self.index, 'type': 'buy'})
            print(f"BUY at {current_close} on index {self.index}")
            self.increase_streak = 0  # Reset streak after buying

    def check_sell_conditions(self, current_close):
        """ Check and Execute Sell Conditions """
        if self.position and self.entry_price is not None:
            # Sell if the price increases consecutively q_candles times
            if self.increase_streak >= self.q_candles:
                self.execute_sell(current_close, "Increase Streak")

            # Sell if the price decreases consecutively x_candles times
            elif self.decrease_streak >= self.x_candles:
                self.execute_sell(current_close, "Decrease Streak")

            # Sell if the loss exceeds the loss threshold
            elif (current_close / self.entry_price - 1) <= -self.loss_threshold:
                self.execute_sell(current_close, "Loss Threshold")

    def execute_sell(self, current_close, reason):
        """ Execute a Sell Order """
        self.sell()
        self.trade_history.append({'index': self.index, 'type': 'sell'})
        print(f"SELL ({reason}) at {current_close} on index {self.index}")

        # Reset state after selling
        self.entry_price = None
        self.position.close()
        self.increase_streak = 0
        self.decrease_streak = 0

    def plot_candlestick(self):
        """ Plot Candlestick Chart with Buy and Sell Trades """
        data = self.data.df[['Open', 'High', 'Low', 'Close']]
        data.index.name = 'Date'

        # Plot the candlestick chart
        fig, ax1 = plt.subplots(figsize=(18, 10))
        mpf.plot(data, type='candle', ax=ax1, style='charles', show_nontrading=True, warn_too_much_data=999999)

        # Plot buy and sell trades
        buy_trades = [trade for trade in self.trade_history if trade['type'] == 'buy']
        sell_trades = [trade for trade in self.trade_history if trade['type'] == 'sell']

        buy_dates = [self.data.index[trade['index']] for trade in buy_trades]
        buy_prices = [self.data.Close[trade['index']] * 0.98 for trade in buy_trades]  # Adjust for commissions
        sell_dates = [self.data.index[trade['index']] for trade in sell_trades]
        sell_prices = [self.data.Close[trade['index']] * 1.02 for trade in sell_trades]  # Adjust for commissions

        # Plot buy trades
        ax1.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy Trades', s=100)
        for date, price in zip(buy_dates, buy_prices):
            ax1.annotate(f"BUY\n{price:.2f}", (date, price), textcoords="offset points", xytext=(0, 10), ha='center', color='green')

        # Plot sell trades
        ax1.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell Trades', s=100)
        for date, price in zip(sell_dates, sell_prices):
            ax1.annotate(f"SELL\n{price:.2f}", (date, price), textcoords="offset points", xytext=(0, -15), ha='center', color='red')

        # Draw lines connecting buy and sell trades
        for buy, sell in zip(buy_trades, sell_trades):
            buy_date = self.data.index[buy['index']]
            sell_date = self.data.index[sell['index']]
            buy_price = self.data.Close[buy['index']]
            sell_price = self.data.Close[sell['index']]
            ax1.plot([buy_date, sell_date], [buy_price, sell_price], color='blue', linestyle='--', linewidth=1)

        # Add legend
        ax1.legend()

        print(self.trade_history)

        return fig

    def plot_equity_curve(self, equity_curve):
        """ Plot the Equity Curve """
        fig, ax2 = plt.subplots(figsize=(18, 10))
        ax2.plot(equity_curve.index, equity_curve, label='Equity Curve')
        ax2.set_ylabel('Equity')
        ax2.legend()

        return fig