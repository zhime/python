import time
from datetime import datetime

date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
print(date)


print(datetime.fromtimestamp(1779159793).strftime("%Y-%m-%d %H:%M:%S"))


print(time.strftime("%Y-%m-%d %H:%M:%S"))
print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))