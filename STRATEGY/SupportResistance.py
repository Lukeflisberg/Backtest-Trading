""" LIBRARIES """
from backtesting import Strategy
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import mplfinance as mpf

""" CLASS """
class strategy(Strategy):
    maximum_candels_to_compute = 100  # Maximum number of candles to consider when calculating average candle size.
    maximum_main_levels = 3  # Maximum number of main support or resistance levels that can exist at any given time.
    lenience_factor = 0.25  # Percentage lenience when determining if a price is close to a support or resistance level.
    sharpness_threshold = 0.005  # Percentage threshold to determine if a price point is steep enough to qualify as support or resistance.
    support_resistance_multiplier = 0.01  # Multiplier to adjust lenience when replacing support or resistance levels.
    strong_indicator_range = 10  # Range within which a level is considered close to a strong indicator.
    maximum_plot_strong_indicators = 3  # Maximum number of strong indicators to plot on the chart.
    top_strong_indicator_percent = 0.25  # Percentage of the strongest indicators to display on the chart.
    support_resistance_radius = 2  # Radius (number of candles) to check around a price point to determine if it is a local support or resistance.

    def init(self):
        self.current_candle_index = -1  # Tracks the index of the current candle being processed.
        self.strong_indicator_levels = []  # Stores strong indicator levels as dictionaries with details like type, value, and usage count.
        self.main_support_levels = []  # Stores the main support levels identified by the strategy.
        self.main_resistance_levels = []  # Stores the main resistance levels identified by the strategy.
        self.tracked_main_levels = []  # Tracks the usage of main support and resistance levels, including first and last calls.
        self.average_candle_sizes = []  # Stores the average size of candles (difference between high and low) for calculations.
        self.support_points_series = []  # Stores all identified support points for plotting purposes.
        self.resistance_points_series = []  # Stores all identified resistance points for plotting purposes.
        self.trade_history = []  # Stores the history of buy and sell trades, including their type and index.
        self.active_buy_positions = []  # Tracks all currently active buy positions.
        self.active_sell_positions = []  # Tracks all currently active sell positions.
        
        # Log statistics
        self.total_profit_loss = 0  # Tracks the total profit/loss in price.
        self.total_profit_loss_percent = 0  # Tracks the total profit/loss in percentage.

    def is_local_support(self, candle_index):
        """ Check if the current price is a local minimum, and that it is steep """        
        threshold = self.data.Low[candle_index] * self.sharpness_threshold
        
        is_local_min = all(self.data.Low[candle_index - j] > self.data.Low[candle_index - j + 1] for j in range(self.support_resistance_radius, 0, -1)) and \
                       all(self.data.Low[candle_index + j] > self.data.Low[candle_index + j - 1] for j in range(1, self.support_resistance_radius + 1))

        is_steep = max(
            abs(self.data.Low[candle_index-1] - self.data.Low[candle_index]),
            abs(self.data.Low[candle_index+1] - self.data.Low[candle_index])
        ) >= threshold

        return is_local_min and is_steep 
    
    def is_local_resistance(self, candle_index):
        """ Check if the current price is a local maximum, and that it is steep """
        threshold = self.data.High[candle_index] * self.sharpness_threshold

        is_local_max = all(self.data.High[candle_index - j] < self.data.High[candle_index - j + 1] for j in range(self.support_resistance_radius, 0, -1)) and \
                       all(self.data.High[candle_index + j] < self.data.High[candle_index + j - 1] for j in range(1, self.support_resistance_radius + 1))

        is_steep = max(
            abs(self.data.High[candle_index-1] - self.data.High[candle_index]),
            abs(self.data.High[candle_index+1] - self.data.High[candle_index])
        ) >= threshold

        return is_local_max and is_steep

    def calculate_average_candle_size(self):
        """ Compute the average difference of the given levels list """
        if len(self.average_candle_sizes) < 1:
            return 0  # Avoid division by zero
        elif len(self.average_candle_sizes) > self.maximum_candels_to_compute: # Return the latest amounts
            return np.mean(self.average_candle_sizes[-self.maximum_candels_to_compute:])
        else: 
            return np.mean(self.average_candle_sizes)
    
    def find_nearest_main_level(self, level_value, main_levels):
        """ Find the nearest main level to a given support/resistance """
        if not main_levels: 
            return None, np.inf
        nearest = min(main_levels, key=lambda x: abs(x-level_value))
        return nearest, abs(nearest - level_value)
    
    def replace_main_level(self, level_value, main_levels, level_type, candle_index):
        """ Replace the closest main support/resistance if conditions are met """
        avg_candle_size = self.calculate_average_candle_size()
        lenience_threshold = avg_candle_size * self.lenience_factor
        
        nearest, diff = self.find_nearest_main_level(level_value, main_levels) 
        
        # Check if nearest is None (i.e., main_levels is empty)
        if nearest is None: 
           # If main_levels is empty, directly add the level as the first main level
           main_levels.append(level_value)
           if level_type == "support":
              self.support_points_series.append((self.data.index[candle_index], level_value))
           elif level_type == "resistance":
              self.resistance_points_series.append((self.data.index[candle_index], level_value))
           self.track_main_level_usage(level_value, candle_index)
           self.update_strong_indicator_levels(level_type, level_value)  
           return

        if diff > lenience_threshold:
            if level_type == "support":
                self.support_points_series.append((self.data.index[candle_index], level_value))
                should_replace = (level_value > nearest * (1 + self.support_resistance_multiplier) or level_value < nearest)
            elif level_type == "resistance":
                self.resistance_points_series.append((self.data.index[candle_index], level_value))
                should_replace = (level_value < nearest * (1 - self.support_resistance_multiplier) or level_value > nearest)

            # Appends main if max is not met
            if len(main_levels) < self.maximum_main_levels:
                main_levels.append(level_value)
            elif should_replace:
                # Replace the closest main level
                # Less lenient for resistances in cases, and less lenient for supports in cases
                main_index = main_levels.index(nearest)
                self.track_main_level_usage(main_levels[main_index], candle_index) # Creates the final_call for the main level before being replaced 
                main_levels[main_index] = level_value

             # Adds new main level to tracking
            self.track_main_level_usage(level_value, candle_index)
            self.update_strong_indicator_levels(level_type, level_value)  # Update strong indicators
    
    def track_main_level_usage(self, level_value, candle_index):
        """ Track usage of main support/resistance levels """
        for item in self.tracked_main_levels:
            if item['value'] == level_value:
                item['nr_calls'] += 1
                item['last_call'] = candle_index
                return
        
        # If level is new, add it to tracking
        self.tracked_main_levels.append({'first_call': candle_index, 'value': level_value, 'nr_calls': 1, 'last_call': -1})
    
    def update_strong_indicator_levels(self, level_type, level_value):
        """ Update the strong indicators list """
        for indicator in self.strong_indicator_levels:
            if abs(indicator['value'] - level_value) <= self.strong_indicator_range:
                indicator['nr_times'] += 1
                return

        # If no matching strong indicator is found, add a new one
        self.strong_indicator_levels.append({'type': level_type, 'value': level_value, 'nr_times': 1})
    
    def update_support_and_resistance(self):
        # Calculate the average candle size
        candle_size = self.data.High[self.current_candle_index] - self.data.Low[self.current_candle_index]
        self.average_candle_sizes.append(candle_size)


        if self.is_local_support(self.current_candle_index - self.support_resistance_radius):
            support_level = self.data.Low[self.current_candle_index - self.support_resistance_radius]
            self.replace_main_level(support_level, self.main_support_levels, "support", self.current_candle_index - self.support_resistance_radius)

        if self.is_local_resistance(self.current_candle_index - self.support_resistance_radius):
            resistance_level = self.data.High[self.current_candle_index - self.support_resistance_radius]
            self.replace_main_level(resistance_level, self.main_resistance_levels, "resistance", self.current_candle_index - self.support_resistance_radius)

    def is_price_near_support(self):    
        price = self.data.Close[self.current_candle_index]
        for support in self.main_support_levels:
            if abs(price - support) <= self.calculate_average_candle_size() * self.lenience_factor:   
                return True
        return False 
    
    def is_price_near_resistance(self):
        price = self.data.Close[self.current_candle_index]
        for resistance in self.main_resistance_levels:
            if abs(price - resistance) <= self.calculate_average_candle_size() * self.lenience_factor:
                return True
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

        if self.current_candle_index < self.support_resistance_radius*2: 
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

        self.update_support_and_resistance() 


        # Bounce trade logic
        if self.is_price_near_support():   
            if self.has_active_sell_position():
                self.close_active_sell_position() 
              
            if not self.has_active_buy_position():
                self.open_new_buy_position() 

        if self.is_price_near_resistance():
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

        # Plot support points series
        if self.support_points_series:
            support_dates, support_levels = zip(*self.support_points_series)
            support_scatter = ax1.scatter(support_dates, support_levels, color='green', label='Support Points', s=15)

        # Plot resistance points series
        if self.resistance_points_series:
            resistance_dates, resistance_levels = zip(*self.resistance_points_series)
            resistance_scatter = ax1.scatter(resistance_dates, resistance_levels, color='red', label='Resistance Points', s=15)

        # Draw dotted lines for main support points
        support_lines = []
        for item in self.tracked_main_levels:
            if item['value'] in [level for _, level in self.support_points_series]:
                first_date = self.data.index[item['first_call']]
                last_date = self.data.index[item['last_call']]
                line, = ax1.plot([first_date, last_date], [item['value'], item['value']], color='green', linestyle='--')
                support_lines.append(line)

        # Draw dotted lines for main resistance points
        resistance_lines = []
        for item in self.tracked_main_levels:
            if item['value'] in [level for _, level in self.resistance_points_series]:
                first_date = self.data.index[item['first_call']]
                last_date = self.data.index[item['last_call']]
                line, = ax1.plot([first_date, last_date], [item['value'], item['value']], color='red', linestyle='--')
                resistance_lines.append(line)

        # Plot buy and sell trades
        buy_trades = [trade for trade in self.trade_history if trade['type'] == 'buy']
        sell_trades = [trade for trade in self.trade_history if trade['type'] == 'sell']

        buy_dates = [self.data.index[trade['index']] for trade in buy_trades]
        buy_prices = [self.data.Close[trade['index']]*0.98 for trade in buy_trades] # Takes commissions into account
        sell_dates = [self.data.index[trade['index']] for trade in sell_trades]
        sell_prices = [self.data.Close[trade['index']]*1.02 for trade in sell_trades] # Takes commissions into account

        buy_scatter = ax1.scatter(buy_dates, buy_prices, color='green', marker='^', label='Buy Trades', s=100)
        sell_scatter = ax1.scatter(sell_dates, sell_prices, color='red', marker='v', label='Sell Trades', s=100)

        # Plot strong indicators
        sorted_indicators = sorted(self.strong_indicator_levels, key=lambda x: (-x['nr_times'], -self.strong_indicator_levels.index(x)))
        num_to_plot = max(1, int(len(sorted_indicators) * self.top_strong_indicator_percent))
        num_to_plot = min(num_to_plot, self.maximum_plot_strong_indicators)
        top_indicators = sorted_indicators[:num_to_plot]

        strong_indicator_values = [indicator['value'] for indicator in top_indicators]
        strong_indicator_lines = ax1.hlines(y=strong_indicator_values, xmin=data.index.min(), xmax=data.index.max(), colors='purple', linestyles='-', label='Strong Indicators')

        # Manualy creating handles are some are not identified
        legend_handles = [
            Line2D([0], [0], color='green', marker='o', linestyle='', label='Support Points'),
            Line2D([0], [0], color='red', marker='o', linestyle='', label='Resistance Points'),
            Line2D([0], [0], color='green', marker='^', linestyle='', label='Buy Trades'),
            Line2D([0], [0], color='red', marker='v', linestyle='', label='Sell Trades'),
            Line2D([0], [0], color='green', linestyle='--', label='Support Lines'),
            Line2D([0], [0], color='red', linestyle='--', label='Resistance Lines'),
            Line2D([0], [0], color='purple', linestyle='-', label='Strong Indicators')
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
            if label == "Support Points":
                support_scatter.set_visible(not support_scatter.get_visible())
            elif label == "Resistance Points":
                resistance_scatter.set_visible(not resistance_scatter.get_visible())
            elif label == "Buy Trades":
                buy_scatter.set_visible(not buy_scatter.get_visible())
            elif label == "Sell Trades":
                sell_scatter.set_visible(not sell_scatter.get_visible())
            elif label == "Support Lines":
                for line in support_lines:
                    line.set_visible(not line.get_visible())
            elif label == "Resistance Lines":
                for line in resistance_lines:
                    line.set_visible(not line.get_visible())
            elif label == "Strong Indicators":
                strong_indicator_lines.set_visible(not strong_indicator_lines.get_visible())

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
