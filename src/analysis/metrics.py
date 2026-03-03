import pandas as pd

from src.constants.config import LOGS_DIR

df = pd.read_csv(str(LOGS_DIR))

# средняя очередь по направлениям
print(df.groupby("dir")["queue"].mean())

# максимальная очередь
print(df.groupby("dir")["queue"].max())

# общее среднее
print("Overall:", df["queue"].mean())
