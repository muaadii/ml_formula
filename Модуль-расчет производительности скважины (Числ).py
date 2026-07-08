from dataclasses import dataclass
import math
import matplotlib.pyplot as plt
import numpy as np


def compute_coeffs(dh, tau, node_count):
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
    pressure = np.full(node_count, p_e, dtype=float)

    pressure_history = np.empty((time_count, node_count), dtype=float)
    well_pressure_history = np.empty(time_count, dtype=float)

    g = q_const / kprod

    pressure_history[0] = pressure
    well_pressure_history[0] = pressure[0]

    for time_index in range(1, time_count):
        pressure_old = pressure.copy()

        xi = np.zeros(node_count, dtype=float)
        eta = np.zeros(node_count, dtype=float)

        alpha_1 = c[1] - a[1]
        beta_1 = -b[1]
        delta_1 = pressure_old[1] - a[1] * g * dh

        if abs(alpha_1) < 1e-30:
            raise ZeroDivisionError(
                "Нулевой знаменатель в начале прямой прогонки."
            )

        xi[1] = -beta_1 / alpha_1
        eta[1] = delta_1 / alpha_1

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

        pressure[i] = eta[i]

        for i in range(node_count - 3, 0, -1):
            pressure[i] = xi[i] * pressure[i + 1] + eta[i]

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


@dataclass(frozen=True)
class Layer:
    name: str
    genitive_name: str
    initial_pressure: float
    outer_radius: float
    thickness: float
    permeability: float
    viscosity: float
    porosity: float
    total_compressibility: float
    rate: float

    @property
    def kappa(self):
        return self.permeability / (
            self.porosity
            * self.viscosity
            * self.total_compressibility
        )

    @property
    def kprod(self):
        return (
            2
            * math.pi
            * self.permeability
            * self.thickness
            / self.viscosity
        )


@dataclass(frozen=True)
class GridParameters:
    node_count: int = 1200
    time_count: int = 800
    total_days: float = 100


@dataclass
class LayerResult:
    radii: np.ndarray
    pressure_history: np.ndarray
    well_pressure_history: np.ndarray


@dataclass
class SimulationResult:
    time_seconds: np.ndarray
    time_days: np.ndarray
    layer_results: list
    layer_rates_m3day: list
    total_rate_m3day: float


class ReservoirSimulation:
    def __init__(self, layers, grid, well_radius=0.05):
        self.layers = layers
        self.grid = grid
        self.well_radius = well_radius

    def calculate_layer(self, layer, dt_sec):
        dh = np.log(layer.outer_radius / self.well_radius) / (
            self.grid.node_count - 1
        )
        tau = dt_sec * layer.kappa / self.well_radius**2
        radii = np.exp(
            np.linspace(
                np.log(self.well_radius),
                np.log(layer.outer_radius),
                self.grid.node_count,
            )
        )
        a, b, c = compute_coeffs(dh, tau, self.grid.node_count)

        pressure_history, well_pressure_history = solve_plast_const_q(
            a=a,
            b=b,
            c=c,
            dh=dh,
            kprod=layer.kprod,
            p_e=layer.initial_pressure,
            q_const=layer.rate,
            node_count=self.grid.node_count,
            time_count=self.grid.time_count,
        )

        return LayerResult(
            radii=radii,
            pressure_history=pressure_history,
            well_pressure_history=well_pressure_history,
        )

    def run(self):
        time_seconds = np.linspace(
            0.0,
            self.grid.total_days * 86400.0,
            self.grid.time_count,
        )
        time_days = time_seconds / 86400.0
        dt_sec = time_seconds[1] - time_seconds[0]

        layer_results = []

        for layer in self.layers:
            print(f"Выполняется расчет для {layer.genitive_name}...")
            layer_results.append(self.calculate_layer(layer, dt_sec))

        layer_rates_m3day = [layer.rate * 86400.0 for layer in self.layers]
        total_rate_m3day = sum(layer_rates_m3day)

        return SimulationResult(
            time_seconds=time_seconds,
            time_days=time_days,
            layer_results=layer_results,
            layer_rates_m3day=layer_rates_m3day,
            total_rate_m3day=total_rate_m3day,
        )


class ResultPlotter:
    def __init__(self, layers, grid, well_radius=0.05):
        self.layers = layers
        self.grid = grid
        self.well_radius = well_radius
        self.target_times = [
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

    def plot_pressure_profiles(self, axis, layer, layer_result, time_days):
        time_indices = [
            int(np.argmin(np.abs(time_days - target_time)))
            for target_time in self.target_times
        ]

        for index in time_indices:
            current_time = time_days[index]
            axis.plot(
                layer_result.radii,
                layer_result.pressure_history[index] / 1e6,
                label=f"{current_time:.1f} сут",
            )

        axis.axhline(
            y=layer.initial_pressure / 1e6,
            color="black",
            linestyle="--",
            alpha=0.5,
            label=(
                "Начальное давление = "
                f"{layer.initial_pressure / 1e6:.0f} МПа"
            ),
        )
        axis.axvline(
            x=self.well_radius,
            color="green",
            linestyle=":",
            alpha=0.5,
            label="Скважина",
        )
        axis.axvline(
            x=layer.outer_radius,
            color="blue",
            linestyle=":",
            alpha=0.5,
            label="Внешняя граница",
        )
        axis.set_title(
            f"{layer.name}: давление при постоянном дебите"
        )
        axis.set_xlabel("Радиус, м")
        axis.set_ylabel("Давление, МПа")
        axis.grid(True, which="both", alpha=0.5)
        axis.legend(loc="best", fontsize=7)

    def plot_well_pressures(self, axis, result):
        positive_time_mask = result.time_days > 0.0

        for layer, layer_result in zip(self.layers, result.layer_results):
            axis.semilogx(
                result.time_days[positive_time_mask],
                layer_result.well_pressure_history[positive_time_mask] / 1e6,
                linewidth=1.5,
                label=layer.name,
            )

        axis.set_title("Изменение забойного давления во времени")
        axis.set_xlabel("Время, сут")
        axis.set_ylabel("Забойное давление, МПа")
        axis.grid(True, which="both", linestyle="--", alpha=0.7)
        axis.legend(loc="best")

    def plot_rates(self, axis, result):
        line_styles = ["-", "--"]

        for index, (layer, rate) in enumerate(
            zip(self.layers, result.layer_rates_m3day)
        ):
            axis.plot(
                result.time_days,
                np.full_like(result.time_days, rate),
                linestyle=line_styles[index],
                linewidth=1.5,
                label=f"{layer.name} = {rate:.1f} м³/сут",
            )

        axis.plot(
            result.time_days,
            np.full_like(result.time_days, result.total_rate_m3day),
            linewidth=2.5,
            label=f"Суммарный = {result.total_rate_m3day:.1f} м³/сут",
        )
        axis.set_title("Дебиты, постоянные и заданные по условию")
        axis.set_xlabel("Время, сут")
        axis.set_ylabel("Дебит, м³/сут")
        axis.grid(True, alpha=0.5)
        axis.legend(loc="best")

    def show(self, result):
        figure, axes = plt.subplots(
            2,
            2,
            figsize=(16, 10),
            constrained_layout=True,
        )

        for axis, layer, layer_result in zip(
            axes[0],
            self.layers,
            result.layer_results,
        ):
            self.plot_pressure_profiles(
                axis,
                layer,
                layer_result,
                result.time_days,
            )

        self.plot_well_pressures(axes[1, 0], result)
        self.plot_rates(axes[1, 1], result)
        plt.show()


class StatisticsPrinter:
    @staticmethod
    def print(result, layers, total_days):
        print("\n=== РЕЗУЛЬТАТЫ РАСЧЕТА ===")

        for layer, rate in zip(layers, result.layer_rates_m3day):
            print(
                f"Заданный дебит {layer.genitive_name}: "
                f"{rate:.1f} м³/сут"
            )

        print(
            f"Заданный суммарный дебит: "
            f"{result.total_rate_m3day:.1f} м³/сут"
        )

        for index, (layer, layer_result) in enumerate(
            zip(layers, result.layer_results)
        ):
            prefix = "\n" if index == 0 else ""
            print(
                f"{prefix}Забойное давление {layer.genitive_name} через "
                f"{total_days:.0f} сут: "
                f"{layer_result.well_pressure_history[-1] / 1e6:.3f} МПа"
            )


def main():
    layers = [
        Layer(
            name="Пласт 1",
            genitive_name="пласта 1",
            initial_pressure=30e6,
            outer_radius=250,
            thickness=8,
            permeability=10e-15,
            viscosity=3e-3,
            porosity=0.15,
            total_compressibility=3e-10,
            rate=0.0003472,
        ),
        Layer(
            name="Пласт 2",
            genitive_name="пласта 2",
            initial_pressure=25e6,
            outer_radius=300,
            thickness=10,
            permeability=10e-15,
            viscosity=2e-3,
            porosity=0.15,
            total_compressibility=3e-10,
            rate=0.0003472,
        ),
    ]
    grid = GridParameters()
    simulation = ReservoirSimulation(layers, grid)
    result = simulation.run()
    StatisticsPrinter.print(result, layers, grid.total_days)
    ResultPlotter(layers, grid).show(result)


if __name__ == "__main__":
    main()