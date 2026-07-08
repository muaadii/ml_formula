import math
import matplotlib.pyplot as plt
import numpy as np


# ==================== ПАРАМЕТРЫ ПЛАСТОВ ====================

r_w = 0.05  # радиус скважины, м

# Пласт 1
p_e1 = 30e6        # начальное пластовое давление, Па
r_e1 = 250         # внешний радиус, м
h_1 = 8            # толщина, м
k_1 = 10e-15       # проницаемость, м²
mu_1 = 3e-3        # вязкость, Па·с
phi_1 = 0.15       # пористость
c_total_1 = 3e-10  # сжимаемость, 1/Па

kappa_1 = k_1 / (phi_1 * mu_1 * c_total_1)
kprod1 = 2 * math.pi * k_1 * h_1 / mu_1

# Пласт 2
p_e2 = 25e6
r_e2 = 300
h_2 = 10
k_2 = 10e-15
mu_2 = 2e-3
phi_2 = 0.15
c_total_2 = 3e-10

kappa_2 = k_2 / (phi_2 * mu_2 * c_total_2)
kprod2 = 2 * math.pi * k_2 * h_2 / mu_2


# ==================== ЗАДАННЫЕ ПОСТОЯННЫЕ ДЕБИТЫ ====================

# При указанных значениях каждый пласт дает примерно 30 м³/сут.
q1_const = 0.0003472  # м³/с
q2_const = 0.0003472  # м³/с


# ==================== ПАРАМЕТРЫ СЕТКИ ====================

N = 1200       # число узлов по радиусу
Nt = 800       # число временных точек, включая t = 0
t_total = 100  # полное время расчета, сут

# Временная сетка включает начальный и конечный моменты.
T_sec = np.linspace(0.0, t_total * 86400.0, Nt)
T_day = T_sec / 86400.0
dt_sec = T_sec[1] - T_sec[0]

# Шаги по логарифмической координате z = ln(r / r_w).
dh1 = np.log(r_e1 / r_w) / (N - 1)
dh2 = np.log(r_e2 / r_w) / (N - 1)

# Безразмерные шаги по времени.
tau1 = dt_sec * kappa_1 / r_w**2
tau2 = dt_sec * kappa_2 / r_w**2

# Радиальные сетки.
r1 = np.exp(np.linspace(np.log(r_w), np.log(r_e1), N))
r2 = np.exp(np.linspace(np.log(r_w), np.log(r_e2), N))


# ==================== ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ====================

def compute_coeffs(dh, tau, node_count):
    """Вычисляет коэффициенты неявной разностной схемы."""
    indices = np.arange(node_count - 1, dtype=float)

    coefficient = (
        np.exp(-2.0 * indices * dh)
        * (np.exp(2.0 * dh) - np.exp(-2.0 * dh))
        / (4.0 * dh)
    )

    a = coefficient * tau / dh**2
    b = a.copy()
    c = 2.0 * a + 1.0

    return a, b, c


def solve_plast_const_q(
    a,
    b,
    c,
    dh,
    kprod,
    p_e,
    q_const,
    node_count,
    time_count,
):
    """
    Решает задачу при постоянном дебите методом прогонки.

    Левая граница:
        P[0] = P[1] - g * dh

    Правая граница:
        P[-1] = P[-2]
    """
    pressure = np.full(node_count, p_e, dtype=float)

    pressure_history = np.empty((time_count, node_count), dtype=float)
    well_pressure_history = np.empty(time_count, dtype=float)

    # Производная давления по z на стенке скважины.
    g = q_const / kprod

    # Начальное состояние соответствует T_day[0] = 0.
    pressure_history[0] = pressure
    well_pressure_history[0] = pressure[0]

    for time_index in range(1, time_count):
        pressure_old = pressure.copy()

        xi = np.zeros(node_count, dtype=float)
        eta = np.zeros(node_count, dtype=float)

        # Первый внутренний узел, i = 1.
        # Используется левое условие Неймана:
        # P[0] = P[1] - g * dh.
        alpha_1 = c[1] - a[1]
        beta_1 = -b[1]
        delta_1 = pressure_old[1] - a[1] * g * dh

        if abs(alpha_1) < 1e-30:
            raise ZeroDivisionError(
                "Нулевой знаменатель в начале прямой прогонки."
            )

        xi[1] = -beta_1 / alpha_1
        eta[1] = delta_1 / alpha_1

        # Внутренние узлы i = 2 ... N - 3.
        for i in range(2, node_count - 2):
            alpha_i = -a[i]
            beta_i = c[i]
            gamma_i = -b[i]
            delta_i = pressure_old[i]

            denominator = alpha_i * xi[i - 1] + beta_i

            if abs(denominator) < 1e-30:
                raise ZeroDivisionError(
                    f"Нулевой знаменатель прогонки в узле i={i}."
                )

            xi[i] = -gamma_i / denominator
            eta[i] = (
                delta_i - alpha_i * eta[i - 1]
            ) / denominator

        # Последний внутренний узел i = N - 2.
        # Правая граница непроницаема:
        # P[N - 1] = P[N - 2].
        i = node_count - 2

        alpha_i = -a[i]
        beta_i = c[i] - b[i]
        delta_i = pressure_old[i]

        denominator = alpha_i * xi[i - 1] + beta_i

        if abs(denominator) < 1e-30:
            raise ZeroDivisionError(
                "Нулевой знаменатель в конце прямой прогонки."
            )

        xi[i] = 0.0
        eta[i] = (
            delta_i - alpha_i * eta[i - 1]
        ) / denominator

        # Обратный ход.
        pressure[i] = eta[i]

        for i in range(node_count - 3, 0, -1):
            pressure[i] = xi[i] * pressure[i + 1] + eta[i]

        # Граничные условия.
        pressure[0] = pressure[1] - g * dh
        pressure[-1] = pressure[-2]

        if not np.all(np.isfinite(pressure)):
            raise FloatingPointError(
                f"На временном шаге {time_index} получены "
                "некорректные значения давления."
            )

        pressure_history[time_index] = pressure
        well_pressure_history[time_index] = pressure[0]

    return pressure_history, well_pressure_history


# ==================== ВЫЧИСЛЕНИЕ КОЭФФИЦИЕНТОВ ====================

a1, b1, c1 = compute_coeffs(dh1, tau1, N)
a2, b2, c2 = compute_coeffs(dh2, tau2, N)


# ==================== РАСЧЕТ ДАВЛЕНИЙ ====================

print("Выполняется расчет для пласта 1...")

P1_hist, pw1_hist = solve_plast_const_q(
    a=a1,
    b=b1,
    c=c1,
    dh=dh1,
    kprod=kprod1,
    p_e=p_e1,
    q_const=q1_const,
    node_count=N,
    time_count=Nt,
)

print("Выполняется расчет для пласта 2...")

P2_hist, pw2_hist = solve_plast_const_q(
    a=a2,
    b=b2,
    c=c2,
    dh=dh2,
    kprod=kprod2,
    p_e=p_e2,
    q_const=q2_const,
    node_count=N,
    time_count=Nt,
)


# ==================== ПОДГОТОВКА ДАННЫХ ====================

q1_m3day = q1_const * 86400.0
q2_m3day = q2_const * 86400.0
q_total_m3day = q1_m3day + q2_m3day

target_times = [
    1.0,
    3.1,
    5.2,
    7.3,
    9.4,
    11.6,
    13.7,
    15.8,
    17.9,
    20.0,
]

time_indices = [
    int(np.argmin(np.abs(T_day - target_time)))
    for target_time in target_times
]

# Для semilogx исключается нулевая точка времени.
positive_time_mask = T_day > 0.0


# ==================== ПОСТРОЕНИЕ ВСЕХ ГРАФИКОВ ====================

figure, axes = plt.subplots(
    2,
    2,
    figsize=(16, 10),
    constrained_layout=True,
)

ax_pressure_1 = axes[0, 0]
ax_pressure_2 = axes[0, 1]
ax_well_pressure = axes[1, 0]
ax_rates = axes[1, 1]


# ---------- 1. Профили давления пласта 1 ----------

for index in time_indices:
    current_time = T_day[index]

    ax_pressure_1.plot(
        r1,
        P1_hist[index] / 1e6,
        label=f"{current_time:.1f} сут",
    )

ax_pressure_1.axhline(
    y=p_e1 / 1e6,
    color="black",
    linestyle="--",
    alpha=0.5,
    label=f"Начальное давление = {p_e1 / 1e6:.0f} МПа",
)

ax_pressure_1.axvline(
    x=r_w,
    color="green",
    linestyle=":",
    alpha=0.5,
    label="Скважина",
)

ax_pressure_1.axvline(
    x=r_e1,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label="Внешняя граница",
)

ax_pressure_1.set_title(
    "Пласт 1: давление при постоянном дебите"
)
ax_pressure_1.set_xlabel("Радиус, м — логарифмическая шкала")
ax_pressure_1.set_ylabel("Давление, МПа")
ax_pressure_1.grid(True, which="both", alpha=0.5)
ax_pressure_1.legend(loc="best", fontsize=7)


# ---------- 2. Профили давления пласта 2 ----------

for index in time_indices:
    current_time = T_day[index]

    ax_pressure_2.plot(
        r2,
        P2_hist[index] / 1e6,
        label=f"{current_time:.1f} сут",
    )

ax_pressure_2.axhline(
    y=p_e2 / 1e6,
    color="black",
    linestyle="--",
    alpha=0.5,
    label=f"Начальное давление = {p_e2 / 1e6:.0f} МПа",
)

ax_pressure_2.axvline(
    x=r_w,
    color="green",
    linestyle=":",
    alpha=0.5,
    label="Скважина",
)

ax_pressure_2.axvline(
    x=r_e2,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label="Внешняя граница",
)

ax_pressure_2.set_title(
    "Пласт 2: давление при постоянном дебите"
)
ax_pressure_2.set_xlabel("Радиус, м — логарифмическая шкала")
ax_pressure_2.set_ylabel("Давление, МПа")
ax_pressure_2.grid(True, which="both", alpha=0.5)
ax_pressure_2.legend(loc="best", fontsize=7)


# ---------- 3. Динамика забойных давлений ----------

ax_well_pressure.semilogx(
    T_day[positive_time_mask],
    pw1_hist[positive_time_mask] / 1e6,
    linewidth=1.5,
    label="Пласт 1",
)

ax_well_pressure.semilogx(
    T_day[positive_time_mask],
    pw2_hist[positive_time_mask] / 1e6,
    linewidth=1.5,
    label="Пласт 2",
)

ax_well_pressure.set_title(
    "Изменение забойного давления во времени"
)
ax_well_pressure.set_xlabel(
    "Время, сут — логарифмическая шкала"
)
ax_well_pressure.set_ylabel("Забойное давление, МПа")
ax_well_pressure.grid(
    True,
    which="both",
    linestyle="--",
    alpha=0.7,
)
ax_well_pressure.legend(loc="best")


# ---------- 4. Постоянные дебиты ----------

ax_rates.plot(
    T_day,
    np.full_like(T_day, q1_m3day),
    linewidth=1.5,
    label=f"Пласт 1 = {q1_m3day:.1f} м³/сут",
)

ax_rates.plot(
    T_day,
    np.full_like(T_day, q2_m3day),
    linestyle="--",
    linewidth=1.5,
    label=f"Пласт 2 = {q2_m3day:.1f} м³/сут",
)

ax_rates.plot(
    T_day,
    np.full_like(T_day, q_total_m3day),
    linewidth=2.5,
    label=f"Суммарный = {q_total_m3day:.1f} м³/сут",
)

ax_rates.set_title(
    "Дебиты, постоянные и заданные по условию"
)
ax_rates.set_xlabel("Время, сут")
ax_rates.set_ylabel("Дебит, м³/сут")
ax_rates.grid(True, alpha=0.5)
ax_rates.legend(loc="best")


# ==================== СТАТИСТИКА ====================

print("\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")
print(
    f"Заданный дебит пласта 1: "
    f"{q1_m3day:.1f} м³/сут"
)
print(
    f"Заданный дебит пласта 2: "
    f"{q2_m3day:.1f} м³/сут"
)
print(
    f"Заданный суммарный дебит: "
    f"{q_total_m3day:.1f} м³/сут"
)

print(
    f"\nЗабойное давление пласта 1 через "
    f"{t_total:.0f} сут: {pw1_hist[-1] / 1e6:.3f} МПа"
)
print(
    f"Забойное давление пласта 2 через "
    f"{t_total:.0f} сут: {pw2_hist[-1] / 1e6:.3f} МПа"
)

plt.show()