# Project Specification - Iterated Prisoner's Dilemma Interaction Graph

> **Status:** Draft proposal, subject to adaptation as research and prototyping continue.
> A separate **technical specification** (repository layout, module design, data formats,
> testing) follows next week and supersedes the implementation detail sketched here.

## The Pitch

Model cooperation in the Iterated Prisoner's Dilemma on a network where ties have a **weight
equal to accumulated interaction history**, and where that weight **decays over time if
unused** — like real relationships fading without contact. Agents compute a **happiness**
score per neighbour (weight × payoff from that relationship) and greedily relocate away from
low-happiness ties toward better ones. Agents also have a fixed **connection style**:
broad-shallow (many weak ties) vs narrow-deep (few strong ties). Question: does one
connection style win out, or do both survive?

## Research Question

> Under a decaying-weight, happiness-driven greedy relocation dynamic, does a population of
> agents with fixed broad-shallow vs narrow-deep tie-allocation phenotypes converge to
> dominance of one phenotype, or reach stable coexistence — and how does this depend on the
> weight decay rate?

## Background Information

### The Prisoner's Dilemma

- A two-player, single-round game, where each player independently chooses to cooperate or defect, with payoffs that typically follow in the matrix:

| | Opp C | Opp D |
|---|---|---|
| You C | 3 | 0 |
| You D | 5 | 1 |

- The typical payoff ordering is `T > R > P > S` (temptation > reward > punishment > sucker's payoff), with the additional constraint `2R > T + S` so that alternating exploitation is worse than sustained cooperation.
- The single-round Nash equilibrium is mutual defection, which is Pareto-inferior — this gap between individually rational and collectively optimal behaviour is the phenomenon the project is interested in.

### The Iterated Prisoner's Dilemma

- The Prisoner's Dilemma game played repeatedly between the same two players, so that a player's current choice can be conditioned on the opponent's past behaviour.
- Repetition makes reciprocal strategies possible - cooperation can become viable given a long-term relationship.
- Typical Strategies (each agent is assigned one at initialisation and keeps it for the whole run):

| Strategy | Behaviour | Nice | Retaliates | Forgives | Role in our model |
|---|---|:--:|:--:|:--:|---|
| **Always Cooperate** (AllC) | Cooperates unconditionally. | ✅ | ❌ | — | Exploitable baseline; measures how much free-riding the network permits. |
| **Always Defect** (AllD) | Defects unconditionally. | ❌ | — | ❌ | Invasion threat; tests whether decay and relocation can quarantine defectors. |
| **Tit-for-Tat** (TFT) | Cooperates first, then copies opponent's previous move. | ✅ | ✅ | ✅ | The Axelrod benchmark; our primary reciprocal reference point. |
| **Generous TFT** (GTFT) | TFT, but forgives a defection with some probability *g*. | ✅ | ✅ | ✅✅ | Breaks retaliation deadlocks; matters once noise is enabled. |
| **Tit-for-Two-Tats** (TF2T) | Defects only after two consecutive defections. | ✅ | ✅ | ✅✅ | More noise-tolerant than TFT, but more exploitable. |
| **Grudger / Grim Trigger** | Cooperates until first defection, then defects forever. | ✅ | ✅✅ | ❌ | Maximally unforgiving; under decay its dead ties should starve and be relocated away. |
| **Pavlov** (Win-Stay-Lose-Shift) | Repeats last move if it scored well (T or R), switches otherwise. | ✅ | ✅ | ✅ | Exploits AllC, recovers from noise; behaves differently to TFT under our weighting. |
| **Random** | Cooperates with probability 0.5. | ❌ | ❌ | — | Control; establishes a noise floor for interpreting results. |

- "Nice" means never the first to defect. The initial strategy mix is a configurable parameter — the interesting comparisons are TFT-vs-AllD invasion, and whether Grudger's unforgivingness is rescued or punished by the ability to relocate.
- Outcomes are sensitive to the environment, typically - the composition of the strategy pool, the presence of noise (misimplemented or misread moves), and — the focus of this project — **who plays whom**.

### Inspiration and Acknowledgements

- The inspiration of this project comes from the Veritasium video *"What Game Theory Reveals About Life, The Universe, and Everything"* (https://www.youtube.com/watch?v=mScpHTIi-kM).
- The video's key takeaways are:
    - Reciprocity beats both pure exploitation and pure altruism.
    - Forgiveness matters once noise is introduced
    - Evolutionary/population dynamics can shift when certain strategies can dominate over time.
- Our extension is structural - Axelrod's tournaments assume a well-mixed population where every agent plays every other agent equally. We instead ask what happens when interaction is mediated by a social network with unequal, evolving tie strengths.
- Underlying game theory is attributed to Flood & Dresher (1950), Tucker's narrative framing, and Axelrod & Hamilton (1981); the video is credited as the accessible presentation that motivated the topic, not as a primary source.


### Weighted Graphs and Modelling

- Agents are modelled as **nodes** in an undirected graph; an **edge** between two nodes represents a social connection along which interactions can occur.
- Each edge carries a **weight equal to accumulated interaction history** — a count of how much contact the pair has had, *not* an abstract affinity score. It rises with every interaction and falls with time spent apart.
- Weight therefore represents relationship *depth*: it determines how much of an agent's limited attention a tie consumes, and how heavily that tie counts in the agent's happiness evaluation.
- This makes structure and behaviour co-evolve: contact builds weight, weight sustains contact, and neglect dissolves the tie entirely — so the topology is an output of the model, not just an input.
- Graph structure is expected to matter in its own right — clustering, degree distribution, and path length plausibly determine whether cooperative clusters can form and resist invasion by defectors.

## Positioning and Contribution

- A literature scan (Huang 2015; Shen 2018; Fu et al. 2009; Xiong et al. 2024; Tomassini, Pestelacci & Luthi; and the tie-decay network literature) found two of our four mechanisms already established individually: **weight × payoff utility** and **greedy partner-switching**. These are deliberately low-risk building blocks we can cite rather than defend.
- Two mechanisms were **not** found combined with these anywhere: **time-based weight decay** (existing decay models are outcome-driven — they punish defection — rather than driven by time since last interaction), and **broad vs narrow tie allocation as a competing trait**.
- Our contribution is therefore the **specific composition**, not a reinvention of the IPD: what happens to cooperation when neglect alone can kill a tie, and when agents differ in how thinly they spread their attention.
- *Action item:* full bibliographic details for the above still need compiling for the report.

## Proposed Structure

### Graph Initialisation

- `N` agents instantiated as nodes, with `N` in the low hundreds to keep round-by-round simulation tractable while still producing network-scale effects.
- Baseline topology from a configurable generator — Erdős–Rényi (random) as the neutral default, with Watts–Strogatz (small-world) and Barabási–Albert (scale-free) retained as options if time allows. Initial topology matters less here than usual, since relocation reshapes the graph over time.
- Each agent is assigned **(a)** a strategy from the table above, and **(b)** a connection phenotype, both from configurable distributions, both fixed for the run.
- Edge weights initialised at a small uniform `w₀` so that no tie starts privileged.

### Connection Phenotypes and the Degree Budget

- Every agent has the same fixed **attention budget** `B` — the number of interactions it can spend per round across all its ties. This is the scarce resource that makes the two styles a genuine trade-off rather than a free choice.
- **Broad-shallow** agents hold up to `k_broad` ties (many); **narrow-deep** agents hold up to `k_narrow` ties (few), with `k_broad > k_narrow`. Budget divides across held ties, so each tie of a broad agent receives roughly `B / k_broad` interactions per round, and each tie of a narrow agent roughly `B / k_narrow`.
- Because reinforcement is per-interaction and decay is per-unit-time, **the equilibrium weight of a tie scales roughly as `B / (k · α)`**. Spreading attention across more ties lowers the steady-state weight of each one.
- This gives the model's central testable prediction: there should be a **critical decay rate above which broad-shallow ties cannot be sustained at all** — decay outpaces reinforcement, weights fall below the severance threshold, and the phenotype collapses. Below that rate, breadth should pay off by diversifying away from bad partners. Locating that threshold is the primary result we are chasing.

### Interactions

- Simulation proceeds in discrete **rounds**; within a round each agent allocates its budget `B` across its current ties, and each allocated unit is one Prisoner's Dilemma game between the endpoints.
- Each agent chooses C or D using its strategy and its **per-neighbour interaction history**, meaning an agent can cooperate with one neighbour while defecting against another.
- Payoffs accumulate into a per-agent score and into a **per-tie payoff record**, which the happiness calculation needs.
- An optional **noise** parameter flips a proportion of actions to test the robustness of unforgiving strategies (Grudger, TFT).

### Changing Weights

- **Reinforcement:** each interaction increments the tie's weight (`w ← w + 1`, or a tunable increment).
- **Decay:** weight decays with time since last contact, `w(t) = w · e^(−α · Δt)`. Applied to every edge each round; since reinforcement is additive this is equivalent to computing it lazily at interaction time, which may be the cheaper implementation.
- **Severance:** ties falling below a threshold `θ` are removed, freeing budget and a slot in the agent's degree allowance. This is how neglect ends a relationship.
- `α` is the model's headline parameter and one of the two primary sweep axes.

### Happiness and Relocation

- Each agent computes a **happiness** score per neighbour combining the tie's weight with the payoff that relationship returns — working definition `happiness = weight × payoff`.
- **Greedy relocation:** an agent compares a tie's happiness against its alternatives and switches away from poor ones, subject to its degree cap. Relocation is per-neighbour and local, not a global re-optimisation.
- Candidate partners are drawn locally (friend-of-a-friend / triadic closure) rather than globally, so relocation stays a plausible social process and the network retains structure.
- Two open design issues need resolving before this is built — how a *prospective* tie is valued when it has no history yet, and whether raw payoff or payoff-above-baseline is the right factor. See [Open Questions](#open-questions).

### Simulation

- Core loop: **allocate budget → play games → accumulate payoffs → reinforce weights → apply decay → sever dead ties → evaluate happiness → relocate → refill free slots**.
- Parameters exposed: `N`, topology, initial strategy mix, phenotype mix, payoff matrix values, `B`, `k_broad`, `k_narrow`, decay rate `α`, severance threshold `θ`, noise level, rounds, and random seed.
- Implementation in Python using `networkx` for graph handling, `numpy` for the numerics, and `matplotlib` for output; runs seeded and repeated so results are reproducible and reported as averages over multiple trials.

### Primary Sweep

- **Axis 1 — decay rate `α`:** from near-zero (ties effectively permanent) to high (only constantly-refreshed ties survive).
- **Axis 2 — phenotype mix ratio:** all-broad / all-narrow / 50-50 and intermediate ratios, to separate "which phenotype wins in competition" from "which does better in isolation".
- Deliberately **two axes only**. Payoff values, degree budget and topology are held at defaults; a full combinatorial sweep is out of scope.

### Analysis and Results Gathering

- **Phenotype outcomes (primary):** mean payoff and survival by phenotype over time, and whether a mixed population converges to dominance or coexistence — the direct answer to the research question.
- **Behavioural metrics:** overall cooperation rate over time, mean payoff per strategy, and cooperation rate broken down by phenotype.
- **Structural metrics:** mean and distribution of tie weights, edge count/density, degree distribution by phenotype, tie turnover/relocation rate, clustering, and assortativity by strategy and phenotype — tracked over time, not just at the end state.
- **Headline output:** a heatmap of phenotype outcome across `α` × mix ratio, to locate the predicted critical decay rate — supported by time-series plots and network visualisations coloured by strategy and phenotype, weighted by tie strength.

## Scope (v1)

Target: **5-6 weeks.**

**Fixed / not evolving:**

- Classical PD strategy per agent — assigned at init, never changes.
- Connection phenotype (broad/narrow) — assigned at init, never changes; we compare populations rather than letting the trait evolve.

**Core mechanisms (build these):**

- Weight decay `w(t) = w · e^(−α·Δt)`, reinforced on interaction.
- Happiness per tie: weight × payoff.
- Greedy per-neighbour relocation.
- Degree/attention budget, without which broad vs narrow is not an actual trade-off.

**Explicitly cut from v1** (stretch goals only if ahead of schedule):

- Strategy evolution/mutation.
- Phenotype switching as emergent behaviour.
- Full combinatorial sweep across payoff values, degree budget, topology, etc.

## Open Questions

1. **Valuing a prospective tie.** If happiness is `weight × payoff` and a new tie starts at weight ≈ 0, its happiness is ≈ 0, so a purely comparative relocation rule would *never* fire — incumbency always wins and the network freezes. Options: make severance absolute (drop any tie below `θ`, then refill the slot) rather than comparative; value candidates on expected payoff alone; or credit prospects with a notional starting weight. **Needs deciding before the relocation code is written.**
2. **Raw payoff vs payoff above baseline.** With all-positive payoffs, a long-standing mutual-*defection* tie (high weight, `P = 1`) can outscore a fresh mutual-*cooperation* tie (low weight, `R = 3`). That sunk-cost behaviour is arguably realistic, but may dominate the dynamics and hand the result to narrow-deep trivially. Payoff *relative to* `P`, or average payoff per interaction with weight as a salience multiplier, are the obvious alternatives.
3. **Calibrating `B`, `k_broad`, `k_narrow` and `θ`** so neither phenotype is trivially non-viable at the default decay rate — the sweep is only informative if both styles are alive at baseline.
4. **Synchronous vs asynchronous updating**, and whether relocation runs every round or on a slower clock than interaction.
5. **Scale of `N`** against runtime, given per-edge updates every round across a two-axis sweep with repeats.

## Further Research Requirements

- **This specification is a proposal and is expected to change.** The happiness function, decay model and relocation rule in particular are likely to be revised once we finish the literature review and run early prototypes.
- Literature to finish reviewing: Axelrod (1984) on tournament results; Nowak & May (1992) on spatial/lattice games; Santos & Pacheco on scale-free networks and cooperation; plus full citations for the co-evolutionary and tie-decay work listed under [Positioning and Contribution](#positioning-and-contribution).
- **Validation plan:** reproduce a known baseline result (e.g. Tit-for-Tat's performance in a well-mixed population) before enabling decay, budget and relocation — so that any novel behaviour can be attributed to our mechanisms rather than to implementation error.
- A separate **technical specification** covering repository layout, module design, data formats, and testing is to be produced in the following week and will supersede the implementation details sketched here.
