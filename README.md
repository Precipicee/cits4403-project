# cits4403-project

## Project Overview

An agent-based model of **simulated cities** whose citizens are **givers** (cooperative
strategies) or **takers** (exploitative ones), playing the **Iterated Prisoner's Dilemma** with
the people they actually meet. Who meets whom is not uniform: it depends on distance through
the network, the strength of an existing relationship, and how popular the other person is.
Relationships strengthen with use, fade without it, break when they get too weak, and are
replaced.

The project asks how the mix of givers and takers shapes a city's **prosperity**, the
**inequality** of that prosperity, and the **in-groups and out-groups** that form.

## Problem and Approach

- **Why agent-based?** Prosperity, inequality and segregation are outcomes no citizen chooses.
  They emerge from many local interactions that are each simple to state, which is what an
  agent-based model is for.
- **What we vary:** the share of givers, from 0 to 1; the starting network (random, small-world,
  scale-free); and how fast relationships decay.
- **How we build it:** six steps, each adding one mechanism to the last, from a well-mixed
  crowd to the full city, and finally to a city whose citizens walk away from anyone who cheats
  them. Each step reproduces the previous one exactly, with the same seed,
  when its new mechanism is switched off.

## Model Description

Each round is one week. For every round:

1. every agent picks a partner, weighing proximity, habit and the partner's popularity;
2. each pair plays one round of the Prisoner's Dilemma, with payoffs `(T, R, P, S) = (5, 3, 1, 0)`;
3. payoffs are added to wealth, which is created by interaction rather than moved between agents;
4. relationships used this week strengthen, unused ones decay, and faded ones break;
5. agents with spare capacity form new relationships, usually with friends of friends;
6. popularity ratings update from how cooperative and well-connected each agent has been.

In the last step (S6), a pair stops meeting once either side has defected on the other, and the
tie between them ends. The rule never mentions class or strategy.

Partner choice uses the attractiveness score `s_ij = (w_ij + eps)^beta * R_j^gamma * d_ij^-lambda`,
where `w` is relationship strength, `R` is the partner's popularity rating and `d` is path length
through the network. Meetings received are capped, because attention is finite.

Every stochastic step draws from one seeded `numpy` generator passed through the model, so a run
is reproducible from its seed and configuration.

## Results

`notebooks/Project.ipynb` runs the whole argument end to end in about fifteen minutes on an
8-worker pool: the model, what we expected, the sweep, the figures, an extended analysis and the
conclusions. Every figure is also written to `figures/` as PDF and PNG.

- **Prosperity:** more givers make a city richer, and without refusal the gain is faster once givers
  are common. Where the gain speeds up depends on the settings.
- **Inequality without refusal is positional:** a city where everyone plays Tit-for-Tat, so that
  nobody is ever exploited, is nearly as unequal as a mixed one (Gini 0.21 against 0.23). The
  inequality comes from who gets met, and the rank order of who gets met is largely settled
  within the first 25 of 500 weeks.
- **Refusal changes whose inequality it is:** when citizens walk away from a defector for even
  one week, the Gini barely moves, but the share of inequality lying between givers and takers
  rises from about a fifth to more than four fifths.
- **Refusal that never forgives becomes exclusion:** almost every taker ends with no ties, givers
  keep about 3,000 each at every mix, and taker-dominated cities become poor and unequal. The city
  starts losing wealth once a refusal lasts longer than about a year.
- **A subtle cheat escapes:** Joss, which cooperates nine games in ten, keeps about nine tenths of
  its wealth under refusal, while AllD and Random lose about three quarters of theirs.
- **Sensitivity:** with each of ten settings lowered and raised in turn, refusal always cuts takers
  off and a city where nobody cheats always stays unequal. The faster gain in prosperity (H1) passes
  its test in 15 of the 20 changed settings, and the equal poverty trap (H4) in 11.

## Setup

### Prerequisites

- Python >= 3.10
- pip

### Installation

```bash
git clone https://github.com/Precipicee/cits4403-project.git
cd cits4403-project
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Run the analysis

```bash
jupyter lab notebooks/Project.ipynb
```

Run the cells top to bottom. Every result is generated on the spot; nothing is read from a saved
results file.

### Explore one city

The notebook's *Explore One City* section has controls for the scenario, the share of givers, how
long a refusal lasts, whether Joss is among the takers, and the seed. It runs one city of 60
citizens over 300 weeks and steps through it week by week. The controls need a running Jupyter
kernel; the same controls can be opened in any notebook:

```python
from utilities.explorer import explore

explore({"S5 full city": ("M4", {}), "S6 refusal": ("M5", {})})
```

### Drive the model yourself

```python
from source.presets import stage
from source.simulation import Simulation

cfg = stage("M4", **{"population.n_agents": 200, "population.giver_fraction": 0.5})
sim = Simulation(cfg, root_seed=20260918, repeat=0, instruments=("core", "structure"))
result = sim.run()
```

`stage` takes `"M0"` to `"M5"`, the six build steps, and any configuration key as a keyword
override, so any modelling choice can be changed from a single call. `utilities.runs.sweep`
runs a grid of cities in parallel and returns one row per run.

### Run a larger experiment

A sweep too large for a notebook is declared in code, so its settings stay visible beside its
results, and run in parallel. Aggregated tables are written to `data/results/`:

```python
from source.experiment import Experiment, run

exp = Experiment(name="cities", base="M4", root_seed=20260917, n_seeds=25,
                 axes={"population.giver_fraction": [0.0, 0.25, 0.5, 0.75, 1.0],
                       "topology.name": ["erdos_renyi", "barabasi_albert"]})
tables = run(exp, workers=8)
```

## Project Structure

```
cits4403-project/
|-- source/            model code: game, strategies, encounter, ties, popularity, rewiring, metrics
|-- utilities/         plotting, statistics, checks, small sweeps, the interactive explorer
|-- notebooks/         Project.ipynb: the analysis, start to finish
|-- figures/           figures and the network animation written by the notebook
|-- project_spec.md    project specification
|-- requirements.txt   dependencies
`-- README.md
```
