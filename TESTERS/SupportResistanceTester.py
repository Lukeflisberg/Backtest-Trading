""" LIBRARIES """
from backtesting import Strategy, Backtest
from backtesting.test import GOOG
import numpy as np
import pandas as pd
from pandas import ExcelWriter
from datetime import datetime
import time
from itertools import product

FILE_PATH = "analysis_data.txt"

""" CLASS """
class strategy(Strategy):
    maximum_candels_to_compute = 100  # Maximum number of candles to consider when calculating average candle size.
    maximum_main_levels = 3  # Maximum number of main support or resistance levels that can exist at any given time.
    lenience_factor = 0.1  # Percentage lenience when determining if a price is close to a support or resistance level.
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
                main_levels[main_index] = level_value

            self.update_strong_indicator_levels(level_type, level_value)  # Update strong indicators
    
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
    
""" RUN MULTIPLE BACKTESTS """
# Define parameter ranges
parameter_ranges = {
    "maximum_main_levels": [3, 5, 7],
    "lenience_factor": [0.025, 0.05, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
    "sharpness_threshold": [0, 0.0025, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03],
    "support_resistance_radius": [2],
}

# Define time frames
time_frames = [
    ("2012-01-01", "2012-12-31"),
    ("2013-01-01", "2013-12-31"),
]

# Generate all parameter combinations
parameter_combinations = list(product(*parameter_ranges.values()))

# Store results in a list
results = []

# Calculate the total number of runs
total_runs = len(time_frames) * len(parameter_combinations)
remaining_runs = total_runs

# Measure the time for one backtest
start_time = time.time()

# Run a single backtest to estimate time per run
df_sample = GOOG
params_sample = parameter_combinations[0]

# Set strategy parameters for the sample run
strategy.maximum_main_levels = params_sample[0]
strategy.lenience_factor = params_sample[1]
strategy.sharpness_threshold = params_sample[2]
strategy.support_resistance_radius = params_sample[3]

# Run the sample backtest
bt_sample = Backtest(df_sample, strategy, cash=10_000, commission=0.002)
bt_sample.run()

# Calculate time per run
time_per_run = time.time() - start_time
estimated_total_time = time_per_run * total_runs

# Display estimated total runtime
print(f"Estimated total runtime: {estimated_total_time / 60:.2f} minutes ({estimated_total_time:.2f} seconds)")

# Ensure GOOG index is in datetime format
GOOG.index = pd.to_datetime(GOOG.index)

excel_filename = "analysis_data.xlsx"

with ExcelWriter(excel_filename, engine='openpyxl', mode='a' if pd.io.common.file_exists(excel_filename) else 'w') as writer:
    for start_date, end_date in time_frames:
        # Filter GOOG data for the time frame
        df = GOOG[(GOOG.index >= start_date) & (GOOG.index <= end_date)]

        if df.empty:
            print(f"Warning: No data available for time frame {start_date} to {end_date}. Skipping...")
            continue

        run_results = []
        for params in parameter_combinations:
            print(f"Running backtest... {remaining_runs} runs left.")
            remaining_runs -= 1

            # Set strategy parameters
            strategy.maximum_main_levels = params[0]
            strategy.lenience_factor = params[1]
            strategy.sharpness_threshold = params[2]
            strategy.support_resistance_radius = params[3]

            # Run backtest
            bt = Backtest(df, strategy, cash=10_000, commission=0.002)
            stats = bt.run()

            # Save results to list
            run_results.append({
                "max_main_levels": params[0],
                "lenience_factor": params[1],
                "sharpness_threshold": params[2],
                "support_resistance_radius": params[3],
                "total_profit": stats['Equity Final [$]'] - 10_000,
                "win_rate": stats['Win Rate [%]'],
                "sharpe_ratio": stats['Sharpe Ratio'],
            })

        # Convert to DataFrame
        df_results = pd.DataFrame(run_results)
        now = datetime.now().strftime("%H-%M-%S_%d-%m-%Y")
        sheet_name = f"{now}_{start_date}_{end_date}_SupportRes"
        sheet_name = sheet_name[:31]  # Excel sheet name limit
        df_results.to_excel(writer, sheet_name=sheet_name, index=False)

print(f"\nAll results written to {excel_filename}")