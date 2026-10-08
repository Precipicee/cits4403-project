"""One city, run on demand and stepped through week by week, with controls for the scenario and
its main settings. Needs a running Jupyter kernel; the drawing functions work without one."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from source.presets import stage
from source.simulation import Simulation

from . import plotting as P
from .network_draw import class_legend, draw_snapshot, drifting_layouts
from .runs import DEFAULT_SEED

# Refusal lengths offered for scenarios with refusal; None is a refusal that never ends.
REFUSAL_LENGTHS = {"permanent": None, "1 week": 1, "4 weeks": 4, "13 weeks": 13, "26 weeks": 26,
                   "52 weeks": 52, "104 weeks": 104, "208 weeks": 208}
THREE_TAKERS = {"alld": 1.0, "random": 1.0, "joss": 1.0}


def run_city(base: str, overrides: dict, giver_fraction: float, n_agents: int = 60,
             n_rounds: int = 300, every: int = 10, seed: int = DEFAULT_SEED,
             refusal_memory: int | None = None, joss: bool = False):
    """One city with its week-by-week measures and a snapshot every ``every`` weeks."""
    settings = {**overrides, "population.n_agents": n_agents, "run.n_rounds": n_rounds,
                "population.giver_fraction": giver_fraction, "metrics.snapshot_every": every,
                "encounter.refusal_memory": refusal_memory}
    if joss:
        settings["population.taker_composition"] = THREE_TAKERS
    cfg = stage(base, **settings)
    out = Simulation(cfg, root_seed=seed, repeat=0, instruments=("core", "snapshots")).run().outputs
    return cfg, out["core"], out["snapshots"]


def takers_cut_off(frames: list) -> np.ndarray:
    """Share of takers with no ties at each snapshot; NaN where the city has no takers."""
    out = []
    for snap in frames:
        takers = ~snap["giver"]
        isolated = snap["adjacency"].sum(axis=1) == 0
        out.append(isolated[takers].mean() if takers.any() else np.nan)
    return np.array(out)


def draw_week(core: dict, frames: list, positions: list, k: int, title: str = ""):
    """The city at snapshot ``k`` beside its history up to the end of the run, with the current
    week marked. Scales are fixed over the whole run, so a change between weeks is a real one."""
    snap, week = frames[k], frames[k]["round"]
    weeks = np.arange(1, len(core["gini"]) + 1)
    fig = plt.figure(figsize=(13, 5.6))
    grid = fig.add_gridspec(3, 2, width_ratios=[1.15, 1])
    city = fig.add_subplot(grid[:, 0])
    wealth, gini, shut = (fig.add_subplot(grid[r, 1]) for r in range(3))
    draw_snapshot(city, snap, positions[k], title=f"{title}, week {week}",
                  wealth_scale=(0.0, max(f["wealth"].max() for f in frames)),
                  weight_scale=max(f["weight"].max() for f in frames), by_class=True)
    class_legend(city)
    for cls in ("giver", "taker"):
        wealth.plot(weeks, core[f"wealth_{cls}"], color=P.CLASS_COLOURS[cls], label=f"{cls}s")
    wealth.plot(weeks, core["mean_wealth"], color=P.INK_2, linewidth=1, linestyle="--", label="everyone")
    wealth.set(ylabel="wealth per person")
    wealth.legend(fontsize=8, loc="upper left")
    gini.plot(weeks, core["gini"], color=P.SERIES[6])
    gini.set(ylabel="Gini of wealth", ylim=(0, max(0.3, np.nanmax(core["gini"]) * 1.1)))
    shut.plot([f["round"] for f in frames], takers_cut_off(frames), color=P.CLASS_COLOURS["taker"],
              marker="o", markersize=3)
    shut.set(xlabel="week", ylabel="takers with no ties", ylim=(-0.03, 1.03))
    for ax in (wealth, gini, shut):
        ax.axvline(week, color=P.MUTED, linewidth=1)
        ax.set_xlim(0, weeks[-1])
    return fig


def explore(scenarios: dict, n_agents: int = 60, n_rounds: int = 300, every: int = 10,
            seed: int = DEFAULT_SEED, start: str | None = None):
    """Controls for one city: scenario, giver fraction, refusal length, Joss among the takers and
    seed. *Run* simulates the city; the play button and slider then step through its weeks.

    ``scenarios`` maps a label to ``(stage name, overrides)``, as in the notebook's sweep.
    """
    import ipywidgets as W
    from IPython.display import display

    labels = list(scenarios)
    scenario = W.Dropdown(options=labels, value=start or labels[-1], description="Scenario")
    fraction = W.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1, description="Giver share",
                             readout_format=".1f", continuous_update=False)
    refusal = W.Dropdown(options=list(REFUSAL_LENGTHS), value="permanent", description="Refusal lasts")
    joss = W.Checkbox(value=False, description="Joss among the takers", indent=False)
    seed_box = W.IntText(value=seed, description="Seed")
    run = W.Button(description="Run", button_style="primary")
    status = W.Label()
    # Drawing a week takes about half a second, so frames advance once a second.
    play = W.Play(value=0, min=0, max=0, step=1, interval=1000, description="Play")
    week = W.SelectionSlider(options=[("0", 0)], description="Week", continuous_update=False)
    W.jslink((play, "value"), (week, "index"))
    out = W.Output()
    city = {"busy": False}

    def has_refusal(label):
        base, overrides = scenarios[label]
        return stage(base, **overrides).encounter.refusal != "none"

    def show(k):
        with out:
            out.clear_output(wait=True)
            fig = draw_week(city["core"], city["frames"], city["positions"], k, city["title"])
            plt.show()
            plt.close(fig)

    def simulate(_=None):
        status.value = "Running..."
        base, overrides = scenarios[scenario.value]
        refuses = has_refusal(scenario.value)
        _, core, frames = run_city(base, overrides, fraction.value, n_agents, n_rounds, every,
                                   seed_box.value, REFUSAL_LENGTHS[refusal.value] if refuses else None,
                                   joss.value)
        city.update(core=core, frames=frames, positions=drifting_layouts(frames, seed=3),
                    title=f"{scenario.value.split()[0]}, giver share {fraction.value:.1f}")
        # Swapping the slider's options moves it, so drawing waits until it points at the last week.
        city["busy"] = True
        week.options = [(str(f["round"]), i) for i, f in enumerate(frames)]
        play.max = len(frames) - 1
        week.index = len(frames) - 1
        city["busy"] = False
        status.value = (f"{n_agents} citizens, {n_rounds} weeks, a snapshot every {every} weeks"
                        + ("" if refuses else "; this scenario has no refusal"))
        show(week.index)

    def sync_refusal(change=None):
        refusal.disabled = not has_refusal(scenario.value)

    scenario.observe(sync_refusal, names="value")
    week.observe(lambda change: None if city["busy"] else show(change["new"]), names="index")
    run.on_click(simulate)
    sync_refusal()
    simulate()
    class Panel(W.VBox):
        # The text a static copy of the notebook shows in place of the controls.
        def __repr__(self):
            return "Controls for one city; they need a running Jupyter kernel."

    display(Panel([W.HBox([scenario, fraction, refusal]), W.HBox([joss, seed_box, run, status]),
                   W.HBox([play, week]), out]))
