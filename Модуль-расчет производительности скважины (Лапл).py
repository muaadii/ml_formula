from dataclasses import dataclass
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import factorial, iv, kv


Q_const = 30 / 86400
p_w = 20e6
r_w = 0.05
t_day = 86400


def inv_laplace(func, t, n=12):
    if t <= 0:
        raise ValueError("Время t должно быть больше нуля.")

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


def make_pressure_laplace(r, r_e, kappa, mu, permeability, thickness, rate):
    def pressure_laplace(s):
        alpha = np.sqrt(s / kappa)

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


def make_rate_laplace(rate):
    def rate_laplace(s):
        return rate / s

    return rate_laplace


def check_steady_state(
    rates,
    times,
    layer_name,
    threshold=0.02,
    consecutive_points=3,
):
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


@dataclass(frozen=True)
class Layer:
    name: str
    steady_state_name: str
    initial_pressure: float
    outer_radius: float
    thickness: float
    permeability: float
    viscosity: float
    porosity: float
    total_compressibility: float

    @property
    def kappa(self):
        return self.permeability / (
            self.porosity * self.viscosity * self.total_compressibility
        )


@dataclass
class SimulationResult:
    pressure_times: np.ndarray
    rate_times: np.ndarray
    radii: list
    pressure_curves: list
    layer_rates: list
    total_rates: np.ndarray
    steady_times: list


class ReservoirSimulation:
    def __init__(self, layers, rate, days=20, n_pressure_curves=10):
        self.layers = layers
        self.rate = rate
        self.days = days
        self.n_pressure_curves = n_pressure_curves

    def calculate_pressure_curves(self, layer, pressure_times, radii):
        curves = []

        for t in pressure_times:
            pressures = np.array(
                [
                    layer.initial_pressure
                    + inv_laplace(
                        make_pressure_laplace(
                            r=r,
                            r_e=layer.outer_radius,
                            kappa=layer.kappa,
                            mu=layer.viscosity,
                            permeability=layer.permeability,
                            thickness=layer.thickness,
                            rate=self.rate,
                        ),
                        t,
                        n=12,
                    )
                    for r in radii
                ]
            )
            curves.append(pressures / 1e6)

        return curves

    def calculate_layer_rates(self, rate_times):
        layer_rates = []

        for _ in self.layers:
            rate_laplace = make_rate_laplace(self.rate)
            rates = np.array(
                [
                    max(inv_laplace(rate_laplace, t, n=12) * t_day, 0.0)
                    for t in rate_times
                ]
            )
            layer_rates.append(rates)

        return layer_rates

    def run(self):
        pressure_times = np.linspace(
            t_day,
            self.days * t_day,
            self.n_pressure_curves,
        )
        radii = [
            np.linspace(r_w, layer.outer_radius, 300)
            for layer in self.layers
        ]
        pressure_curves = [
            self.calculate_pressure_curves(layer, pressure_times, layer_radii)
            for layer, layer_radii in zip(self.layers, radii)
        ]

        rate_times = np.linspace(0.9 * t_day, self.days * t_day, 50)
        layer_rates = self.calculate_layer_rates(rate_times)
        total_rates = np.sum(layer_rates, axis=0)
        steady_times = [
            check_steady_state(rates, rate_times, layer.steady_state_name)
            for layer, rates in zip(self.layers, layer_rates)
        ]

        return SimulationResult(
            pressure_times=pressure_times,
            rate_times=rate_times,
            radii=radii,
            pressure_curves=pressure_curves,
            layer_rates=layer_rates,
            total_rates=total_rates,
            steady_times=steady_times,
        )


class ResultPlotter:
    def __init__(self, layers, days):
        self.layers = layers
        self.days = days

    def plot_pressure(self, axis, layer, radii, pressure_times, curves):
        for t, pressures in zip(pressure_times, curves):
            axis.plot(
                radii,
                pressures,
                label=f"t = {t / t_day:.1f} сут",
            )

        axis.axhline(
            layer.initial_pressure / 1e6,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=(
                f"Начальное давление = "
                f"{layer.initial_pressure / 1e6:.0f} МПа"
            ),
        )
        axis.axhline(
            p_w / 1e6,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Забойное давление = {p_w / 1e6:.0f} МПа",
        )
        axis.axvline(
            r_w,
            color="green",
            linestyle=":",
            alpha=0.5,
            label="Скважина",
        )
        axis.axvline(
            layer.outer_radius,
            color="blue",
            linestyle=":",
            alpha=0.5,
            label="Внешняя граница",
        )
        axis.set_title(f"{layer.name}: распределение давления")
        axis.set_xlabel("Радиус, м")
        axis.set_ylabel("Давление, МПа")
        axis.grid(True)
        axis.legend(loc="best", fontsize=7)

    def plot_rates(self, axis, result):
        axis.plot(
            result.rate_times / t_day,
            result.total_rates,
            "k-",
            linewidth=3,
            label="Суммарный дебит",
        )
        axis.plot(
            result.rate_times / t_day,
            result.layer_rates[0],
            "b-",
            linewidth=1.5,
            alpha=0.7,
            label="Дебит пласта 1",
        )
        axis.plot(
            result.rate_times / t_day,
            result.layer_rates[1],
            "r--",
            linewidth=1.5,
            alpha=0.7,
            label="Дебит пласта 2",
        )
        axis.set_title("Динамика дебитов")
        axis.set_xlabel("Время, сут")
        axis.set_ylabel("Дебит, м³/сут")
        axis.grid(which="major", linewidth=0.8, alpha=0.5)
        axis.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.3)
        axis.minorticks_on()
        axis.legend(loc="best")

    def show(self, result):
        figure = plt.figure(figsize=(16, 10), constrained_layout=True)
        grid = figure.add_gridspec(2, 2)
        pressure_axes = [
            figure.add_subplot(grid[0, 0]),
            figure.add_subplot(grid[0, 1]),
        ]
        rates_axis = figure.add_subplot(grid[1, :])

        for axis, layer, radii, curves in zip(
            pressure_axes,
            self.layers,
            result.radii,
            result.pressure_curves,
        ):
            self.plot_pressure(
                axis,
                layer,
                radii,
                result.pressure_times,
                curves,
            )

        self.plot_rates(rates_axis, result)
        plt.show()


class StatisticsPrinter:
    def __init__(self, days):
        self.days = days

    def print(self, result):
        print("\n=== СТАТИСТИКА ДЕБИТОВ ===")
        print(
            f"Начальный суммарный дебит: "
            f"{result.total_rates[0]:.1f} м³/сут"
        )
        print(
            f"  в том числе пласт 1: "
            f"{result.layer_rates[0][0]:.1f} м³/сут"
        )
        print(
            f"  в том числе пласт 2: "
            f"{result.layer_rates[1][0]:.1f} м³/сут"
        )
        print(
            f"\nСуммарный дебит через {self.days} сут: "
            f"{result.total_rates[-1]:.1f} м³/сут"
        )
        print(
            f"  в том числе пласт 1: "
            f"{result.layer_rates[0][-1]:.1f} м³/сут"
        )
        print(
            f"  в том числе пласт 2: "
            f"{result.layer_rates[1][-1]:.1f} м³/сут"
        )


def main():
    layers = [
        Layer(
            name="Пласт 1",
            steady_state_name="пласта 1",
            initial_pressure=30e6,
            outer_radius=250,
            thickness=8,
            permeability=10e-15,
            viscosity=3e-3,
            porosity=0.15,
            total_compressibility=3e-10,
        ),
        Layer(
            name="Пласт 2",
            steady_state_name="пласта 2",
            initial_pressure=25e6,
            outer_radius=300,
            thickness=10,
            permeability=10e-15,
            viscosity=2e-3,
            porosity=0.15,
            total_compressibility=3e-10,
        ),
    ]

    simulation = ReservoirSimulation(
        layers=layers,
        rate=Q_const,
        days=20,
        n_pressure_curves=10,
    )
    result = simulation.run()
    StatisticsPrinter(simulation.days).print(result)
    ResultPlotter(layers, simulation.days).show(result)


if __name__ == "__main__":
    main()