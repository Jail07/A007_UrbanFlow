import pandas as pd
import matplotlib.pyplot as plt
import os


LOGS_TO_COMPARE = {
    "Fixed ": "data/logs/log_fixed.csv",
    "Algorythm": "data/logs/algorythm_logs.csv",
    "PressLight": "data/logs/presslight_log.csv",
    "RL (DQN v1)": "data/logs/urbanflow_detailed_log_1.csv",
    "RL (PPO v2)": "data/logs/urbanflow_detailed_log_2.csv",
    "RL (PPO v4)": "data/logs/urbanflow_detailed_log_04.csv"
}


def compare_multiple_simulations():
    data_frames = {}

    for model_name, path in LOGS_TO_COMPARE.items():
        if not os.path.exists(path):
            print(f"[ВНИМАНИЕ] Файл не найден и будет пропущен: {path}")
            continue

        try:
            df = pd.read_csv(path)
            pivot = df.pivot_table(index="time", columns="intersection", values="queue", aggfunc='sum')
            pivot_smoothed = pivot.rolling(window=10, min_periods=1).mean()
            data_frames[model_name] = pivot_smoothed
        except Exception as e:
            print(f"Ошибка при чтении {path}: {e}")

    if not data_frames:
        print("Нет данных для отображения. Проверьте пути к файлам.")
        return

    all_intersections = set()
    for pivot in data_frames.values():
        all_intersections.update(pivot.columns)
    intersections = sorted(list(all_intersections))

    num_intersections = len(intersections)

    fig, axes = plt.subplots(num_intersections, 1, figsize=(14, 4 * num_intersections))
    fig.suptitle("Сравнение эффективности моделей управления (Динамика очередей)", fontsize=16, fontweight='bold')

    if num_intersections == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    linestyles = ['-', '--', '-.', ':']

    # 4. Отрисовка графиков
    for i, intersection in enumerate(intersections):
        ax = axes[i]

        for j, (model_name, pivot_df) in enumerate(data_frames.items()):
            if intersection in pivot_df.columns:
                ax.plot(pivot_df.index, pivot_df[intersection],
                        label=model_name,
                        color=colors[j % len(colors)],
                        linestyle=linestyles[j % len(linestyles)],
                        linewidth=2.5, alpha=0.9)

        ax.set_title(f"Перекресток: {intersection}", fontsize=12)
        ax.set_ylabel("Машин в пробке")
        if i == num_intersections - 1:
            ax.set_xlabel("Время симуляции (сек)")

        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.tight_layout(rect=[0, 0.02, 0.85, 0.96])
    plt.show()


if __name__ == "__main__":
    compare_multiple_simulations()