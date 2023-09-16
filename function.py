import time
from functools import wraps
from colorama import init, Fore
import openpyxl as xl
init()

workbook = xl.Workbook()
sheet = workbook.active
sheet.title = "処理時間"
column_names = ["関数名", "処理時間"]
sheet.append(column_names)

def stop_watch(func):
    @wraps(func)
    def wrapper(*args, **kargs):
        start = time.time() # 関数読み出し時刻
        result = func(*args, **kargs)
        elapsed_time = time.time() - start
        if f"{func.__name__}" == "get_approximate_failure_rate":
            print(Fore.YELLOW + f"近似計算総処理時間:【{elapsed_time}秒】")
        elif f"{func.__name__}" == "get_actual_failure_rate":
            print(Fore.GREEN + f"真値計算総処理時間:【{elapsed_time}秒】")
        elif f"{func.__name__}" == "get_fastest_approximate_failure_rate":
            print(Fore.BLUE + f"最短計算総処理時間:【{elapsed_time}秒】")
        else:
            print(f"{func.__name__}処理時間：【{elapsed_time}秒】")
        return result
    return wrapper

# 関数の使用例