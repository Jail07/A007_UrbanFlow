import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- НАСТРОЙКИ ПУТЕЙ К ЛОГ-ФАЙЛАМ ---
# Укажите здесь пути к вашим CSV файлам
FIXED_LOG_PATH = "data/logs/log_fixed.csv"  # Лог работы обычного светофора
AI_LOG_PATH = "data/logs/urbanflow_detailed_log.csv"  # Лог работы нейросети


def compare_simulations():
    try:
        # Загружаем данные
        df_fixed = pd.read_csv(FIXED_LOG_PATH)
        df_ai = pd.read_csv(AI_LOG_PATH)

        # Группируем очереди по времени и перекресткам
        pivot_fixed = df_fixed.pivot_table(index="time", columns="intersection", values="queue", aggfunc='sum')
        pivot_ai = df_ai.pivot_table(index="time", columns="intersection", values="queue", aggfunc='sum')

        # Создаем фигуру с 2 графиками (по одному на каждый перекресток)
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        fig.suptitle("Сравнение эффективности: Обычные светофоры vs Нейросеть (MAS)", fontsize=16)

        # Вытаскиваем названия перекрестков из колонок (например, 'Vefa_244500423', 'Kulatov_244500424')
        intersections = pivot_ai.columns

        colors_fixed = ['#ff9999', '#ffcc99']  # Бледные цвета для фиксированного
        colors_ai = ['#cc0000', '#cc6600']  # Яркие цвета для ИИ

        for i, intersection in enumerate(intersections):
            if i > 1: break  # Рисуем максимум 2 графика

            # Линия обычного светофора (пунктир)
            if intersection in pivot_fixed.columns:
                axes[i].plot(pivot_fixed.index, pivot_fixed[intersection],
                             label='Фиксированный тайминг', color=colors_fixed[i], linestyle='--', linewidth=2)

            # Линия ИИ (сплошная)
            axes[i].plot(pivot_ai.index, pivot_ai[intersection],
                         label='Управление ИИ (RL)', color=colors_ai[i], linewidth=2.5)

            axes[i].set_title(f"Динамика очередей: Перекресток {intersection}")
            axes[i].set_ylabel("Количество машин в пробке")
            axes[i].set_xlabel("Время симуляции (сек)")
            axes[i].legend()
            axes[i].grid(True)

            # Закрашиваем разницу (показываем, где ИИ сэкономил время)
            if intersection in pivot_fixed.columns:
                axes[i].fill_between(pivot_ai.index, pivot_ai[intersection], pivot_fixed[intersection],
                                     where=(pivot_fixed[intersection] > pivot_ai[intersection]),
                                     interpolate=True, color='green', alpha=0.1, label='Выигрыш ИИ')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    except FileNotFoundError as e:
        print(f"Ошибка: Не найден файл лога. Убедитесь, что пути указаны верно.\nДетали: {e}")
    except Exception as e:
        print(f"Произошла ошибка при построении графиков: {e}")


if __name__ == "__main__":
    compare_simulations()