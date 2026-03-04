import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd

from src.constants.config import DF_PATH

df = pd.read_csv(str(DF_PATH))

pivot = df.pivot(index="time", columns="dir", values="queue")

pivot.plot()
plt.ylabel("Queue length")
plt.title("Queues over time")
plt.show()
