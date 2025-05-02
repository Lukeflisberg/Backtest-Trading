""" LIBRARIES """
from backtesting import Strategy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf

""" CLASS """
class strategy(Strategy):
    """ STRATEGY PARAMETERS """
    lookback_period = 10  # Number of candles to look back for average body size.

    def init(self):
        """ INTIALIZE STRATEGY """
        self.current_candle_index = -1  # Tracks the index of the current candle being processed.
        self.bullish_bearish = []  # Stores all bullish and bearish points for plotting purposes.
        self.trade_history = []  # Stores the history of buy and sell trades, including their type and index.
        self.active_buy_positions = []  # Tracks all currently active buy positions.
        self.active_sell_positions = []  # Tracks all currently active sell positions.
        self.body_sizes = []  # Stores the average size of candles (difference between high and low) for calculations.
        
        # Log statistics
        self.total_profit_loss = 0  # Tracks the total profit/loss in price.
        self.total_profit_loss_percent = 0  # Tracks the total profit/loss in percentage.

    def detect_fvg(self, lookback_period=lookback_period, body_multiplier=1.5):
        """ Detects Fair Value Gaps (FVGs) in historical price data. """
        first_high = self.data.High[self.current_candle_index-2]
        first_low = self.data.Low[self.current_candle_index-2]
        middle_open = self.data.Open[self.current_candle_index-1]
        middle_close = self.data.Close[self.current_candle_index-1]
        third_low = self.data.Low[self.current_candle_index]
        third_high = self.data.High[self.current_candle_index]

        # Calculate the average absolute body size over the lookback period
        if len(self.body_sizes) >= lookback_period: 
            avg_body_size = np.mean(self.body_sizes[-lookback_period:])
        # If there aren't enough candles, use the largest body size from the available candles
        else: 
            avg_body_size = np.mean(self.body_sizes)

        # Ensure avg_body_size is nonzero to avoid false positives
        avg_body_size = avg_body_size if avg_body_size > 0 else 0.001

        middle_body = abs(middle_close - middle_open)

        # Check for Bullish FVG
        if third_low > first_high and middle_body > avg_body_size * body_multiplier:
            self.bullish_bearish.append({'index': self.current_candle_index, 'type': 'bullish'})   

        # Check for Bearish FVG
        elif third_high < first_low and middle_body > avg_body_size * body_multiplier:
            self.bullish_bearish.append({'index': self.current_candle_index, 'type': 'bearish'})   
        
        else:
            self.bullish_bearish.append({'index': self.current_candle_index, 'type': 'none'})   

    def is_bullish(self):
        """Check if the last detected FVG is bullish."""
        if self.bullish_bearish:
            return self.bullish_bearish[-1]['type'] == "bullish"
        return False
    
    def is_bearish(self):
        if self.bullish_bearish:
            return self.bullish_bearish[-1]['type'] == "bearish"
        return False

    def has_active_sell_position(self):
        return self.position and self.position.is_short
    
    def has_active_buy_position(self):
        return self.position and self.position.is_long

    def close_active_sell_position(self):
        if self.position and self.position.is_short: 
            self.position.close() 
            if self.active_sell_positions: 
                self.active_sell_positions.pop()
    
    def close_active_buy_position(self):
        if self.position and self.position.is_long:
            exit_price = self.data.Close[self.current_candle_index]
            entry_price = self.trade_history[-1]['entry_price']
            profit_loss = exit_price - entry_price
            profit_loss_percent = (profit_loss / entry_price) * 100
            self.total_profit_loss += profit_loss
            self.total_profit_loss_percent += profit_loss_percent
            self.trade_history[-1].update({'exit_price': exit_price, 'profit_loss': profit_loss, 'profit_loss_percent': profit_loss_percent})
            self.position.close()

            if self.active_buy_positions:
                self.active_buy_positions.pop()
    
    def open_new_buy_position(self):
        if not (self.position and self.position.is_long):
            entry_price = self.data.Close[self.current_candle_index]
            self.buy(size=1)
            self.active_buy_positions.append(1)
            self.trade_history.append({'index': self.current_candle_index, 'type': 'buy', 'entry_price': entry_price})
    
    def open_new_sell_position(self):
        if not (self.position and self.position.is_short):
            self.sell(size=1)
            self.active_sell_positions.append(1)   
            self.trade_history.append({'index': self.current_candle_index, 'type': 'sell'})   

    def next(self):
        """ Called on every new candle to identify new levels and update main supports/resistances """
        self.current_candle_index += 1

        # Calculate the average candle size
        body_size = abs(self.data.Close[self.current_candle_index] - self.data.Open[self.current_candle_index])
        self.body_sizes.append(body_size)

        if self.current_candle_index < 2: 
            return  # Avoid accessing out-of-range indexes
        
        # Log candle information
        candle_data = (
            f"Candle {self.current_candle_index}: "
            f"Date={self.data.index[self.current_candle_index]}, "
            f"Open={self.data.Open[self.current_candle_index]}, "
            f"High={self.data.High[self.current_candle_index]}, "
            f"Low={self.data.Low[self.current_candle_index]}, "
            f"Close={self.data.Close[self.current_candle_index]}\n"
        )

        self.detect_fvg()

        # The Trade Logic Is Badddd <20% win rate
        if self.is_bullish():   
            if self.has_active_sell_position():
                self.close_active_sell_position() 
        
            if not self.has_active_buy_position():
                self.open_new_buy_position() 

        if self.is_bearish():
            if self.has_active_buy_position():
                self.close_active_buy_position() 
             
            if not self.has_active_sell_position():
                self.open_new_sell_position()    

    def plot_candlestick(self):
        """ Plot all collected support and resistance levels on a candlestick chart """
        data = self.data.df[['Open', 'High', 'Low', 'Close']]
        data.index.name = 'Date'
        
        # Plot the candelstick
        fig, ax1 = plt.subplots(figsize=(18, 10))
        mpf.plot(data, type='candle', ax=ax1, style='charles', show_nontrading=True)

        # Plot bullish and bearish series
        bullish_point = [point for point in self.bullish_bearish if point['type'] == 'bullish']
        bearish_point = [point for point in self.bullish_bearish if point['type'] == 'bearish']

        bullish_dates = [self.data.index[point['index']] for point in bullish_point]
        bullish_prices = [self.data.Close[point['index']] for point in bullish_point]
        bearish_dates = [self.data.index[point['index']] for point in bearish_point]
        bearish_prices = [self.data.Close[point['index']] for point in bearish_point]

        bullish_scatter = ax1.scatter(bullish_dates, bullish_prices, color='green', marker='o', label='Bullish Points', s=100)
        bearish_scatter = ax1.scatter(bearish_dates, bearish_prices, color='red', marker='o', label='Bearish Points', s=100)

        # Plot buy and sell trades
        buy_trades = [trade for trade in self.trade_history if trade['type'] == 'buy']
        sell_trades = [trade for trade in self.trade_history if trade['type'] == 'sell']
        
        buy_dates = [self.data.index[trade['index']] for trade in buy_trades]
        buy_prices = [self.data.Close[trade['index']]*0.98 for trade in buy_trades] # Takes commissions into account
        sell_dates = [self.data.index[trade['index']] for trade in sell_trades]
        sell_prices = [self.data.Close[trade['index']]*1.02 for trade in sell_trades] # Takes commissions into account

        buy_scatter = ax1.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy Trades', s=100)
        sell_scatter = ax1.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell Trades', s=100)
        
        # Manualy creating handles are some are not identified
        legend_handles = [
            Line2D([0], [0], color='green', marker='o', linestyle='', label='Bullish Points'),
            Line2D([0], [0], color='red', marker='o', linestyle='', label='Bearish Points'),
            Line2D([0], [0], color='green', marker='^', linestyle='', label='Buy Trades'),
            Line2D([0], [0], color='red', marker='v', linestyle='', label='Sell Trades'),
        ]
        legend = ax1.legend(handles=legend_handles)

        # Increase pick radius
        for legend_item in legend.legend_handles:
            legend_item.set_picker(10)

        # Function to toggle visibility
        def toggle_visibility(event):
            legend_item = event.artist
            label = legend_item.get_label()

            # Toggle visibility of the corresponding plot elements
            if label == "Buy Trades":
                buy_scatter.set_visible(not buy_scatter.get_visible())
            elif label == "Sell Trades":
                sell_scatter.set_visible(not sell_scatter.get_visible())
            elif label == "Bullish Point":
                bullish_scatter.set_visible(not bullish_scatter.get_visible())
            elif label == "Bearish Point":
                bearish_scatter.set_visible(not bearish_scatter.get_visible())

            fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', toggle_visibility)
        
        return fig

    def plot_equity_curve(self, equity_curve):
        """ Plot the equity curve """
        fig, ax2 = plt.subplots(figsize=(18, 10))
        ax2.plot(equity_curve.index, equity_curve, label='Equity Curve')
        ax2.set_ylabel('Equity')
        ax2.legend()

        return fig