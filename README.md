# 📊 Backtest Trader
A program to backtest multiple dataframes with multiple strategies and recieve multiple outputs

---

## Features
- 📂 File selection from the `DATA` folder
    >> Add your data files to this folder before or during runtime(currently onlt support .txt files)
- 📑 Strategy selection from the `STRATEGIES` folder
    >> Simply place the python script in this folder
    >> Fork the format of this script:
    ![image](https://github.com/user-attachments/assets/0344d54b-0611-4f2f-ac8a-e929f7113914)
    >> You can change the return statement to return None
- 🐞 Debug screen for logs
- 📈 Graphical backtesting results

---

## 🔧 Dependencies
pip install -...

### 1️⃣ File Selection Window
![image](https://github.com/user-attachments/assets/6053824c-3e2b-4015-be4a-f4a8c3639ebd)

### 1️⃣ Strategy Selection Window
![image](https://github.com/user-attachments/assets/9ecadc12-00cf-4d67-aef1-2e5e1ede2335)

### 1️⃣ Debug Window
![image](https://github.com/user-attachments/assets/56df9991-4e5c-4d40-ad95-5ea8ec6b1bc8)

### 1️⃣ Results Window
![image](https://github.com/user-attachments/assets/80378e6f-991b-42c9-80b5-664d0f39d223)

```plaintext
ToDo:
1. Add compatibility with multiple file types (e.g. csv)
2. Add cookie to store selected data files and strategies for launch
3. Adjust UI
4. Add hidden list for strategies
5. Fix graphs having dumb numbers
