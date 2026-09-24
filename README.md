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
- **How we build it:** five steps, each adding one mechanism to the last, from a well-mixed
  crowd to the full city. Each step reproduces the previous one exactly, with the same seed,
  when its new mechanism is switched off.

## Model Description

Each round is one week. For every round:

1. every agent picks a partner, weighing proximity, habit and the partner's popularity;
2. each pair plays one round of the Prisoner's Dilemma, with payoffs `(T, R, P, S) = (5, 3, 1, 0)`;
3. payoffs are added to wealth, which is created by interaction rather than moved between agents;
4. relationships used this week strengthen, unused ones decay, and faded ones break;
5. agents with spare capacity form new relationships, usually with friends of friends;
6. popularity ratings update from how cooperative and well-connected each agent has been.

Partner choice uses the attractiveness score `s_ij = (w_ij + eps)^beta * R_j^gamma * d_ij^-lambda`,
where `w` is relationship strength, `R` is the partner's popularity rating and `d` is path length
through the network. Meetings received are capped, because attention is finite.

Every stochastic step draws from one seeded `numpy` generator passed through the model, so a run
is reproducible from its seed and configuration.

## Results

`notebooks/Project.ipynb` runs the whole argument end to end in about ten minutes: the model,
what we expected, the sweep, the figures, an extended analysis and the conclusions. In short: more
givers make a city richer, and the gain accelerates once givers pass about 30% of the population,
but the richest cities are not the most equal. Inequality peaks in mixed cities, and a city where
nobody is ever exploited is no more equal than a mixed one, because the inequality is positional.

The extended analysis takes that last point further. Holding everyone on Tit-for-Tat, so that no
game can ever be exploited, total wealth is fixed by the rules and only its distribution varies:
the Gini moves from 0.05 under uniform partner choice to 0.21 under the full rule, and the rank
order of who gets met is settled within the first 25 of 500 weeks. It also decomposes the model
ladder step by step and tracks clustering and connectivity over time. Rewiring a city from a
lattice to a random graph, which collapses clustering elevenfold while holding the population and
the mean degree fixed, moves the cooperation rate by -0.0055 (se 0.0025): real, but some twenty
times less than partner choice moves it in a single step. Within this model, how ties are
arranged is a rounding error next to who gets chosen.

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

### Drive the model yourself

```python
from source.presets import stage
from source.simulation import Simulation

cfg = stage("M4", **{"population.n_agents": 200, "population.giver_fraction": 0.5})
sim = Simulation(cfg, root_seed=20260918, repeat=0, instruments=("core", "structure"))
result = sim.run()
```

`stage` takes `"M0"` to `"M4"`, the five build steps, and any configuration key as a keyword
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
|-- utilities/         plotting, statistics, checks, small sweeps
|-- notebooks/         Project.ipynb: the analysis, start to finish
|-- project_spec.md    project specification
|-- requirements.txt   dependencies
`-- README.md
```
