import sys
import os
import shutil
import importlib.util
import time
import inspect
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from backtesting import Strategy, Backtest
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSizePolicy, QTabWidget,
    QListWidget, QListWidgetItem, QFileDialog, QLabel, QScrollBar, QStackedWidget, QTextEdit, QScrollArea
)
from PyQt5.QtCore import Qt, QTimer

DATA_DIR = "DATA"
STRATEGY_DIR = "STRATEGY"

initial_cash = 10000
commission = 0.02

class FileSelectionWindow(QWidget):
    def __init__(self, next_callback):
        super().__init__()
        self.setWindowTitle("File Selection")
        self.selected_files = set()
        self.delete_mode = False
        self.next_callback = next_callback
        self.toggle_state = False

        # Layouts
        main_layout = QHBoxLayout()
        file_layout = QVBoxLayout()
        button_layout = QVBoxLayout()

        # File list
        self.file_list = QListWidget()
        self.file_list.setSelectionMode(QListWidget.MultiSelection)
        self.file_list.itemClicked.connect(self.toggle_item)
        self.load_files()
        file_layout.addWidget(self.file_list)

        # Buttons
        self.add_btn = QPushButton("Add")
        self.add_btn.clicked.connect(self.add_file)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self.enter_delete_mode)

        self.confirm_btn = QPushButton("Confirm")
        self.confirm_btn.clicked.connect(self.confirm_delete)
        self.confirm_btn.hide()

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.exit_delete_mode)
        self.cancel_btn.hide()

        self.toggle_btn = QPushButton("Toggle")
        self.toggle_btn.clicked.connect(self.toggle_all_files)

        self.next_btn = QPushButton("Next")
        self.next_btn.clicked.connect(self.go_to_next)

        button_layout.addWidget(self.add_btn)
        button_layout.addWidget(self.remove_btn)
        button_layout.addWidget(self.confirm_btn)
        button_layout.addWidget(self.cancel_btn)
        button_layout.addWidget(self.toggle_btn)  
        button_layout.addStretch()
        button_layout.addWidget(self.next_btn)

        main_layout.addLayout(file_layout, 4)
        main_layout.addLayout(button_layout, 1)
        self.setLayout(main_layout)

    def toggle_all_files(self):
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            item.setCheckState(Qt.Checked if self.toggle_state else Qt.Unchecked)

        self.toggle_state = not self.toggle_state  # Flip the toggle state

    def load_files(self):
        check_states = {self.file_list.item(i).text(): self.file_list.item(i).checkState()
                    for i in range(self.file_list.count())}

        self.file_list.clear()

        if not os.path.exists(DATA_DIR):
            os.makedirs(DATA_DIR)

        for file in os.listdir(DATA_DIR):
            item = QListWidgetItem(file)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)

            if file in check_states:
                item.setCheckState(check_states[file])
            else:
                item.setCheckState(Qt.Checked)

            self.file_list.addItem(item)

    def toggle_item(self, item):
        if not self.delete_mode:
            if item.checkState() == Qt.Checked:
                self.selected_files.add(item.text())
            else:
                self.selected_files.discard(item.text())

    def add_file(self):
        file_paths, _ = QFileDialog.getOpenFileNames(self, "Select Files")

        for file in file_paths:
            if file:
                shutil.copy(file, DATA_DIR)

        self.load_files()

    def enter_delete_mode(self):
        self.delete_mode = True
        self.remove_btn.hide()
        self.confirm_btn.show()
        self.cancel_btn.show()

    def exit_delete_mode(self):
        self.delete_mode = False
        self.confirm_btn.hide()
        self.cancel_btn.hide()
        self.remove_btn.show()
        self.load_files()

    def confirm_delete(self):
        selected_items = self.file_list.selectedItems()

        for item in selected_items:
            try:
                os.remove(os.path.join(DATA_DIR, item.text()))
            except FileNotFoundError:
                pass

        self.exit_delete_mode()

    def go_to_next(self):
        toggled_files = [self.file_list.item(i).text() for i in range(self.file_list.count())
                         if self.file_list.item(i).checkState() == Qt.Checked]
        self.next_callback(toggled_files)

class StrategySelectionWindow(QWidget):
    def __init__(self, prev_callback, next_callback):
        super().__init__()
        self.setWindowTitle("Strategy Selection")
        self.prev_callback = prev_callback
        self.next_callback = next_callback

        # Layouts
        main_layout = QVBoxLayout()
        button_layout = QHBoxLayout()

        # Strategy list
        self.strategy_list = QListWidget()
        self.load_strategies()

        # Buttons
        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.go_to_prev)

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_to_next)

        button_layout.addStretch()
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        main_layout.addWidget(self.strategy_list)
        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

    def load_strategies(self):
        self.strategy_list.clear()

        if not os.path.exists(STRATEGY_DIR):
            os.makedirs(STRATEGY_DIR)

        for filename in os.listdir(STRATEGY_DIR):
            item = QListWidgetItem(filename)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self.strategy_list.addItem(item)

    def go_to_prev(self):
        self.prev_callback()

    def go_to_next(self):
        selected_strategies = [self.strategy_list.item(i).text() for i in range(self.strategy_list.count())
                               if self.strategy_list.item(i).checkState() == Qt.Checked]
        self.next_callback(selected_strategies)

class DebugProcessingWindow(QWidget):
    def __init__(self, prev_callback, next_callback, selected_files, selected_strategies):
        super().__init__()
        self.setWindowTitle("Debug & Backtest Processing")
        self.selected_files = selected_files
        self.selected_strategies = selected_strategies
        self.prev_callback = prev_callback
        self.next_callback = next_callback

        self.time_start = {'total': None, 'processing': None, 'backtesting': None}
        self.task_index = 1
        self.list_df = []
        self.results = [{} for _ in range(len(selected_strategies))]

        # Layouts
        main_layout = QVBoxLayout(self)
        button_layout = QHBoxLayout()

        # Log Box
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)

        # Buttons
        self.prev_button = QPushButton("Prev")
        self.prev_button.clicked.connect(self.go_to_prev)
        self.prev_button.hide()

        self.next_button = QPushButton("Next")
        self.next_button.clicked.connect(self.go_to_next)
        self.next_button.hide()

        main_layout.addWidget(self.log_box)
        button_layout.addWidget(self.prev_button)
        button_layout.addWidget(self.next_button)

        main_layout.addLayout(button_layout)
        self.setLayout(main_layout)

        self.time_start['total'] = time.time()
        self.log("Starting processing...")
        self.log(f"Selected files: {self.selected_files}")
        self.log(f"Selected strategies: {self.selected_strategies}")

        # Process files
        self.time_start['processing'] = time.time()
        self.log("")
        self.log("Processing files...")

        self.process_files(self.selected_files)

        self.log("Finished processing files")
        self.log(f"Elapsed time: {time.time() - self.time_start['processing']:.2f}s")

        # Run backtests
        self.time_start['backtesting'] = time.time()
        self.log("")
        self.log("Running backtests...")
        
        self.run_backtests(self.list_df, self.selected_strategies, self.selected_files)

        self.log("Finished Backtesting")
        self.log(f"Elapsed time: {time.time() - self.time_start['backtesting']:.2f}s")

        # Finish processing
        self.log("")
        self.log("Processing complete.")
        self.log(f"TOTAL elapsed time: {time.time() - self.time_start['total']:.2f}s")
        self.next_button.show()
        self.prev_button.show()
            
    def log(self, message):
        self.log_box.append(f"{self.task_index}. {message}")
        self.log_box.verticalScrollBar().setValue(self.log_box.verticalScrollBar().maximum())
        self.task_index += 1

    def process_files(self, selected_files):
        for file in selected_files:
            try:
                time_start = time.time()

                self.log(f"Processing {file}...")
                file_path = os.path.join(DATA_DIR, file)

                # Load the data
                df = pd.read_csv(file_path, 
                                    names=["TICKER", "PER", "DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL", "OPENINT"],
                                    delimiter=",")
                
                # Data cleaning
                df = df[~df["DATE"].astype(str).str.contains("<") & ~df["TIME"].astype(str).str.contains("<")]
                df["DATE"] = df["DATE"].astype(str).str.strip()
                df["TIME"] = df["TIME"].astype(str).str.strip().str.zfill(6)
                df["DATETIME_RAW"] = df["DATE"].astype(str) + df["TIME"].astype(str)
                df["DATETIME"] = pd.to_datetime(df["DATETIME_RAW"], format="%Y%m%d%H%M%S", errors="coerce")
                df.set_index("DATETIME", inplace=True)
                
                df.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close", "VOL": "Volume"}, inplace=True)
                df = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors='coerce')
                df.sort_index(inplace=True)

                if df.isnull().values.any():
                    self.log(f"Error: Data in {file_path} contains missing or invalid values!")
                    return
                
                self.list_df.append(df)
                self.log(f"Processed {file} in {time.time() - time_start:.2f}s")

            except Exception as e:
                self.log(f"Error processing {file}: {e}")
                self.list_df.append(pd.DataFrame())  # Append an empty DataFrame
                continue

    def run_backtests(self, list_df, selected_strategies, selected_files):
        for i, df in enumerate(list_df):
            for j, strategy_file in enumerate(selected_strategies):
                start_time = time.time()
                
                self.log(f"Running backtest for {strategy_file}...")

                # Dynamically load the strategy class from the script
                strategy_path = os.path.join(STRATEGY_DIR, strategy_file)
                module_name = os.path.splitext(strategy_file)[0] 

                try:
                    spec = importlib.util.spec_from_file_location(module_name, strategy_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    # Find the `strategy` class in the dynamically loaded module
                    strategy_class = None
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if name == "strategy" and issubclass(obj, Strategy):
                            strategy_class = obj
                            break

                    if not strategy_class:
                        self.log(f"Error: No valid 'strategy' class found in {strategy_file}")
                        continue

                    # Run the backtest
                    bt = Backtest(df, strategy_class, cash=initial_cash, commission=commission)
                    stats = bt.run()

                    # Save the results
                    self.results[j][selected_files[i]] = stats
                    
                    self.log(f"Backtest {strategy_file} completed in {time.time() - start_time:.2f}s")
                
                except Exception as e:
                    self.log(f"Error running backtest for {strategy_file}: {e}")
                    continue

    def go_to_prev(self):
        self.prev_callback()

    def go_to_next(self):
        self.next_callback(self.results)

class ResultsWindow(QWidget):
    def __init__(self, prev_callback, results, selected_files, selected_strategies):
        super().__init__()
        self.setWindowTitle("Backtest Results")
        self.prev_callback = prev_callback
        self.results = results
        self.selected_files = selected_files
        self.selected_strategies = selected_strategies

        # layouts
        main_layout = QVBoxLayout(self)

        # Back button
        back_btn = QPushButton("Back")
        back_btn.clicked.connect(self.go_to_prev)
        main_layout.addWidget(back_btn)

        # Top-level tab widget for strategies
        strategy_tabs = QTabWidget()
        main_layout.addWidget(strategy_tabs)

        # Create tabs for each strategy
        for j, strategy in enumerate(self.selected_strategies):
            strategy_tab = QTabWidget()
            strategy_layout = QVBoxLayout(strategy_tab)       

            # Nested tab widget for dataframes
            dataframe_tabs = QTabWidget()
            strategy_layout.addWidget(dataframe_tabs)

            # Create tabs for each dataframe
            for i, file in enumerate(self.selected_files):
                dataframe_tab = QWidget()
                dataframe_layout = QVBoxLayout(dataframe_tab)

                # Nested tab widget for graph types
                graph_tabs = QTabWidget()
                dataframe_layout.addWidget(graph_tabs)

                # Candlestick graph tab
                candlestick_tab = QWidget()
                candlestick_layout = QVBoxLayout(candlestick_tab)

                # Equity curve tab
                equity_curve_tab = QWidget()
                equity_curve_layout = QVBoxLayout(equity_curve_tab)

                # Stats tab
                stats_tab = QWidget()
                stats_layout = QVBoxLayout(stats_tab)
                
                # Add tabs to the graph_tabs widget
                graph_tabs.addTab(candlestick_tab, "Candlestick")
                graph_tabs.addTab(equity_curve_tab, "Equity Curve")
                graph_tabs.addTab(stats_tab, "Stats")

                # Delay the initialization of the graphs
                def initialize_ui():
                    try:
                        # Initialize candlestick graph
                        candlestick_canvas = self.create_candlestick_graph(self.results[j][file])
                        candlestick_layout.addWidget(candlestick_canvas)
                    except Exception as e:
                        print(f"Error creating candlestick graph: {e}")
                        placeholder = QLabel("Candlestick graph could not be generated.")
                        candlestick_layout.addWidget(placeholder)

                    try:
                        # Initialize equity curve graph
                        equity_curve_canvas = self.create_equity_curve_graph(self.results[j][file])
                        equity_curve_layout.addWidget(equity_curve_canvas)
                    except Exception as e:
                        print(f"Error creating equity curve graph: {e}")
                        placeholder = QLabel("Equity curve graph could not be generated.")
                        equity_curve_layout.addWidget(placeholder)

                    try:
                        # Initialize results text
                        stats_box = QTextEdit(self.create_stats_text(self.results[j][file]))
                        stats_box.setReadOnly(True)
                        stats_layout.addWidget(stats_box)
                    except Exception as e:
                        print(f"Error creating results text: {e}")
                        placeholder = QLabel("Results text could not be generated.")
                        stats_layout.addWidget(placeholder)
                        
                # Add graph tabs to dataframe tab
                dataframe_layout.addWidget(graph_tabs)
                dataframe_tabs.addTab(dataframe_tab, file)

            # Delay the graph initialization to ensure widgets are fully set up
            QTimer.singleShot(0, initialize_ui)

            # Add dataframe tabs to strategy tab
            strategy_layout.addWidget(dataframe_tabs)
            strategy_tabs.addTab(strategy_tab, strategy)

    def create_candlestick_graph(self, results):
        strategy_instance = results._strategy
        
        fig = strategy_instance.plot_candlestick()

        canvas = FigureCanvas(fig)
        return canvas

    def create_equity_curve_graph(self, results):
        equity_curve = results['_equity_curve']['Equity']
        strategy_instance = results._strategy
        
        fig = strategy_instance.plot_equity_curve(equity_curve)
        canvas = FigureCanvas(fig)
        return canvas

    def create_stats_text(self, results):
        return results.to_string()

    def go_to_prev(self):
        self.prev_callback()

def run_app():
    app = QApplication(sys.argv)
    stacked = QStackedWidget()

    def forward_from_1(selected_files):
        def back_from_2():
            stacked.setCurrentWidget(file_selection_window)

        def forward_from_2(selected_strategies):            
            def back_from_3():
                stacked.setCurrentWidget(strategy_selection_window)

            def forward_from_3(results):
                def back_from_4():
                    stacked.setCurrentWidget(file_selection_window)
                
                results_window = ResultsWindow(prev_callback=back_from_4, results=results, selected_files=selected_files, selected_strategies=selected_strategies)
                stacked.addWidget(results_window)
                stacked.setCurrentWidget(results_window)

            debug_processing_window = DebugProcessingWindow(prev_callback=back_from_3, next_callback=forward_from_3, selected_files=selected_files, selected_strategies=selected_strategies)
            stacked.addWidget(debug_processing_window)
            stacked.setCurrentWidget(debug_processing_window)

        strategy_selection_window = StrategySelectionWindow(prev_callback=back_from_2, next_callback=forward_from_2)
        stacked.addWidget(strategy_selection_window)
        stacked.setCurrentWidget(strategy_selection_window)

    file_selection_window = FileSelectionWindow(next_callback=forward_from_1)
    stacked.addWidget(file_selection_window)
    stacked.setCurrentWidget(file_selection_window)

    stacked.resize(1200, 800)
    stacked.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run_app()