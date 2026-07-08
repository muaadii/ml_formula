import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial, iv, kv


# -------------------- Исходные данные --------------------

# В текущей постановке Q_const применяется отдельно к каждому пласту.
# Поэтому каждый пласт дает 30 м³/сут, а суммарный дебит равен 60 м³/сут.
Q_const = 30 / 86400  # м³/с

p_e1 = 30e6  # Па, начальное давление пласта 1
p_e2 = 25e6  # Па, начальное давление пласта 2
p_w = 20e6   # Па, забойное давление — только опорная линия на графике

r_w = 0.05   # м, радиус скважины
r_e1 = 250   # м, внешняя граница пласта 1
r_e2 = 300   # м, внешняя граница пласта 2

# Параметры первого пласта
h_1 = 8
k_1 = 10e-15
mu_1 = 3e-3
phi_1 = 0.15
c_total_1 = 3e-10
kappa_1 = k_1 / (phi_1 * mu_1 * c_total_1)

# Параметры второго пласта
h_2 = 10
k_2 = 10e-15
mu_2 = 2e-3
phi_2 = 0.15
c_total_2 = 3e-10
kappa_2 = k_2 / (phi_2 * mu_2 * c_total_2)


# -------------------- Обратное преобразование Лапласа --------------------

def inv_laplace(func, t, n=12):
    """Обратное преобразование Лапласа методом Стефеста."""
    if t <= 0:
        raise ValueError("Время t должно быть больше нуля.")

    # Для метода Стефеста число коэффициентов должно быть четным.
    if n % 2 != 0:
        n += 1

    n2 = n // 2
    coefficients = np.zeros(n, dtype=float)

    for i in range(1, n + 1):
        sum_term = 0.0
        k_min = (i + 1) // 2
        k_max = min(i, n2)

        for k in range(k_min, k_max + 1):
            numerator = (k ** n2) * factorial(2 * k)
            denominator = (
                factorial(n2 - k)
                * factorial(k)
                * factorial(k - 1)
                * factorial(i - k)
                * factorial(2 * k - i)
            )
            sum_term += numerator / denominator

        coefficients[i - 1] = sum_term * (-1) ** (n2 + i)

    ln2_t = np.log(2.0) / t
    result = 0.0

    for i in range(1, n + 1):
        s_i = i * ln2_t
        value = np.real_if_close(func(s_i))

        if not np.isfinite(value):
            raise FloatingPointError(
                f"При s={s_i} получено некорректное значение: {value}"
            )

        result += coefficients[i - 1] * float(value)

    return result * ln2_t


# -------------------- Давление в пространстве Лапласа --------------------

def make_pressure_laplace(r, r_e, kappa, mu, permeability, thickness, rate):
    """
    Возвращает функцию давления в пространстве Лапласа
    для заданного радиуса и параметров пласта.
    """
    def pressure_laplace(s):
        alpha = np.sqrt(s / kappa)

        # Матрица граничных условий:
        # 1. На внешней границе dp/dr = 0.
        # 2. На скважине задан постоянный дебит.
        matrix = np.array(
            [
                [iv(1, r_e * alpha), -kv(1, r_e * alpha)],
                [iv(1, r_w * alpha), -kv(1, r_w * alpha)],
            ],
            dtype=float,
        )

        right_side = np.array(
            [
                0.0,
                rate
                * mu
                / (
                    2
                    * np.pi
                    * permeability
                    * thickness
                    * r_w
                    * s
                    * alpha
                ),
            ],
            dtype=float,
        )

        try:
            c1, c2 = np.linalg.solve(matrix, right_side)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(
                f"Не удалось решить систему при r={r}, s={s}"
            ) from error

        return c1 * iv(0, r * alpha) + c2 * kv(0, r * alpha)

    return pressure_laplace


# -------------------- Дебит в пространстве Лапласа --------------------

def make_rate_laplace(rate):
    """
    Для постоянного дебита q(t)=rate:
    преобразование Лапласа равно rate/s.
    """
    def rate_laplace(s):
        return rate / s

    return rate_laplace


# -------------------- Расчет данных --------------------

t_day = 86400
days = 20
n_pressure_curves = 10

pressure_times = np.linspace(t_day, days * t_day, n_pressure_curves)

r_array1 = np.linspace(r_w, r_e1, 300)
r_array2 = np.linspace(r_w, r_e2, 300)

pressure_curves_1 = []
pressure_curves_2 = []

for t in pressure_times:
    pressures_1 = np.array(
        [
            p_e1
            + inv_laplace(
                make_pressure_laplace(
                    r=r,
                    r_e=r_e1,
                    kappa=kappa_1,
                    mu=mu_1,
                    permeability=k_1,
                    thickness=h_1,
                    rate=Q_const,
                ),
                t,
                n=12,
            )
            for r in r_array1
        ]
    )

    pressures_2 = np.array(
        [
            p_e2
            + inv_laplace(
                make_pressure_laplace(
                    r=r,
                    r_e=r_e2,
                    kappa=kappa_2,
                    mu=mu_2,
                    permeability=k_2,
                    thickness=h_2,
                    rate=Q_const,
                ),
                t,
                n=12,
            )
            for r in r_array2
        ]
    )

    # Для удобства давления сразу переводятся в МПа.
    pressure_curves_1.append(pressures_1 / 1e6)
    pressure_curves_2.append(pressures_2 / 1e6)


# Расчет дебитов
rate_times = np.linspace(0.9 * t_day, days * t_day, 50)

rate_laplace_1 = make_rate_laplace(Q_const)
rate_laplace_2 = make_rate_laplace(Q_const)

q1_rates = np.array(
    [
        max(inv_laplace(rate_laplace_1, t, n=12) * t_day, 0.0)
        for t in rate_times
    ]
)

q2_rates = np.array(
    [
        max(inv_laplace(rate_laplace_2, t, n=12) * t_day, 0.0)
        for t in rate_times
    ]
)

total_rates = q1_rates + q2_rates


def check_steady_state(
    rates,
    times,
    layer_name,
    threshold=0.02,
    consecutive_points=3,
):
    """
    Считает режим установившимся, если относительное изменение
    меньше threshold несколько интервалов подряд.
    """
    rates = np.asarray(rates, dtype=float)
    times = np.asarray(times, dtype=float)

    if len(rates) < consecutive_points + 1:
        print(f"Для {layer_name} недостаточно точек.")
        return None

    denominator = np.maximum(np.abs(rates[:-1]), 1e-12)
    relative_change = np.abs(np.diff(rates)) / denominator

    for i in range(len(relative_change) - consecutive_points + 1):
        interval = relative_change[i : i + consecutive_points]

        if np.all(interval < threshold):
            steady_time = times[i + 1]
            print(
                f"Режим {layer_name} близок к установившемуся "
                f"примерно через {steady_time / t_day:.1f} сут."
            )
            return steady_time

    print(
        f"Режим {layer_name} не стабилизировался "
        f"в пределах расчетного интервала."
    )
    return None


steady_time_1 = check_steady_state(
    q1_rates,
    rate_times,
    "пласта 1",
)

steady_time_2 = check_steady_state(
    q2_rates,
    rate_times,
    "пласта 2",
)


# -------------------- Построение всех графиков --------------------

figure = plt.figure(figsize=(16, 10), constrained_layout=True)
grid = figure.add_gridspec(2, 2)

ax_pressure_1 = figure.add_subplot(grid[0, 0])
ax_pressure_2 = figure.add_subplot(grid[0, 1])
ax_rates = figure.add_subplot(grid[1, :])


# График давления пласта 1
for t, pressures in zip(pressure_times, pressure_curves_1):
    ax_pressure_1.plot(
        r_array1,
        pressures,
        label=f"t = {t / t_day:.1f} сут",
    )

ax_pressure_1.axhline(
    p_e1 / 1e6,
    color="black",
    linestyle="--",
    alpha=0.5,
    label=f"Начальное давление = {p_e1 / 1e6:.0f} МПа",
)
ax_pressure_1.axhline(
    p_w / 1e6,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Забойное давление = {p_w / 1e6:.0f} МПа",
)
ax_pressure_1.axvline(
    r_w,
    color="green",
    linestyle=":",
    alpha=0.5,
    label="Скважина",
)
ax_pressure_1.axvline(
    r_e1,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label="Внешняя граница",
)

ax_pressure_1.set_title("Пласт 1: распределение давления")
ax_pressure_1.set_xlabel("Радиус, м")
ax_pressure_1.set_ylabel("Давление, МПа")
ax_pressure_1.grid(True)
ax_pressure_1.legend(loc="best", fontsize=7)


# График давления пласта 2
for t, pressures in zip(pressure_times, pressure_curves_2):
    ax_pressure_2.plot(
        r_array2,
        pressures,
        label=f"t = {t / t_day:.1f} сут",
    )

ax_pressure_2.axhline(
    p_e2 / 1e6,
    color="black",
    linestyle="--",
    alpha=0.5,
    label=f"Начальное давление = {p_e2 / 1e6:.0f} МПа",
)
ax_pressure_2.axhline(
    p_w / 1e6,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Забойное давление = {p_w / 1e6:.0f} МПа",
)
ax_pressure_2.axvline(
    r_w,
    color="green",
    linestyle=":",
    alpha=0.5,
    label="Скважина",
)
ax_pressure_2.axvline(
    r_e2,
    color="blue",
    linestyle=":",
    alpha=0.5,
    label="Внешняя граница",
)

ax_pressure_2.set_title("Пласт 2: распределение давления")
ax_pressure_2.set_xlabel("Радиус, м")
ax_pressure_2.set_ylabel("Давление, МПа")
ax_pressure_2.grid(True)
ax_pressure_2.legend(loc="best", fontsize=7)


# График дебитов
ax_rates.plot(
    rate_times / t_day,
    total_rates,
    "k-",
    linewidth=3,
    label="Суммарный дебит",
)
ax_rates.plot(
    rate_times / t_day,
    q1_rates,
    "b-",
    linewidth=1.5,
    alpha=0.7,
    label="Дебит пласта 1",
)
ax_rates.plot(
    rate_times / t_day,
    q2_rates,
    "r--",
    linewidth=1.5,
    alpha=0.7,
    label="Дебит пласта 2",
)

ax_rates.set_title("Динамика дебитов")
ax_rates.set_xlabel("Время, сут")
ax_rates.set_ylabel("Дебит, м³/сут")
ax_rates.grid(which="major", linewidth=0.8, alpha=0.5)
ax_rates.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
ax_rates.minorticks_on()
ax_rates.legend(loc="best")


# -------------------- Статистика --------------------

print("\n=== СТАТИСТИКА ДЕБИТОВ ===")
print(f"Начальный суммарный дебит: {total_rates[0]:.1f} м³/сут")
print(f"  в том числе пласт 1: {q1_rates[0]:.1f} м³/сут")
print(f"  в том числе пласт 2: {q2_rates[0]:.1f} м³/сут")

print(
    f"\nСуммарный дебит через {days} сут: "
    f"{total_rates[-1]:.1f} м³/сут"
)
print(f"  в том числе пласт 1: {q1_rates[-1]:.1f} м³/сут")
print(f"  в том числе пласт 2: {q2_rates[-1]:.1f} м³/сут")


plt.show()