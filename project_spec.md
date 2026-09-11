# Project Specification: Cooperation, Prosperity and Inequality in Simulated Cities

## 1. The Phenomenon We Are Modelling

Communities of comparable size end up in strikingly different places. Some generate a lot of
wealth and spread it broadly; others generate less, and what they do generate pools in a few
hands. A long-standing explanation is **social capital**: the disposition of people in a
community to cooperate with, rather than exploit, the people they deal with.

The evidence for that explanation is real but, on closer inspection, strongly
context-dependent:

1. **Trust and growth are associated, but not uniformly:** A panel of 64 countries over
   1977-2017 finds that social trust affects long-run growth, predominantly through
   productivity [1]. Yet a 2025 meta-analysis of 957 estimates from 83 studies
   finds the *average* effect of social capital on growth is close to zero, with effects
   ranging from large negative to large positive, heterogeneity that observable study
   characteristics cannot explain, and publication bias inflating reported estimates
   [2]. Cooperation pays off richly in some settings and barely at all in others,
   and the empirical literature cannot yet say why.
2. **Which ties matter, not just how many:** Using 21 billion Facebook friendships, Chetty
   et al. find that *economic connectedness* (the share of high-status friends among
   low-status people) is among the strongest predictors of upward income mobility yet
   identified, whereas social *cohesion* (tight, clique-like friendship groups) is not
   strongly associated with it [3]. The structure of who meets whom carries
   information that aggregate measures of sociability miss.
3. **Inequality feeds back on cooperation:** Panel evidence links higher income inequality to
   lower generalised trust [4], [5], with trust shaped most by the
   stratification of society as a whole [6]. In repeated games, extreme
   inequality prevents cooperation outright [7]. So the *distribution* of
   prosperity is not a side-effect; it shapes whether prosperity can continue.
4. **Whether cooperation reduces inequality is contested:** Network game models report
   cooperation and inequality moving in opposite directions [8], and reputational
   information, which promotes cooperation, tends to reduce payoff dispersion [9].
   But across 33 laboratory experiments, clustered networks produce more inequality than
   random ones and foster segregation between rich and poor [10], and making
   wealth visible amplifies initial inequality [11].

Points 1 and 2 motivate the question; points 3 and 4 make it worth simulating. An empirical
regression cannot hold a community's composition fixed while varying who meets whom. A
model can.

**The abstraction:** A city is a network of people. Each person is an agent with a fixed
disposition, loosely a **giver** or a **taker**, playing repeated Prisoner's Dilemma
interactions with the people they actually encounter. Who you encounter is not uniform: it
depends on how close someone is to you in the social graph, how strong your existing tie is,
and how sought-after they are. Interactions generate wealth, which accrues to individuals.
Aggregate prosperity, its distribution, and the city's social structure are therefore
*outputs* of the mix of dispositions and the structure of encounter, not inputs.

## 2. Research Question and Hypotheses

> **Research question:** In simulated cities that differ in their mix of cooperative and
> exploitative agents, what patterns emerge in (a) aggregate prosperity, (b) the distribution
> of that prosperity across citizens, and (c) social structure: do in-groups and out-groups
> form, and around what?

Four falsifiable hypotheses follow. Each names the observation that would refute it.

| # | Hypothesis | Refuted if |
|---|---|---|
| **H1** | **Prosperity threshold:** Mean wealth per capita rises with the giver fraction *non-linearly*, with a critical fraction below which cooperator clusters cannot form and the city collapses to near-universal defection. | Mean wealth is linear in giver fraction, or shows no threshold. |
| **H2** | **Prosperity and equality decouple:** Existing network models find inequality falling as cooperation rises [8], [9]. We hypothesise that this relationship breaks at high giver fractions: the Gini coefficient of wealth is *not* monotonically decreasing in giver fraction, but flattens or rises at the top end. The proposed mechanism is that cooperation on networks is sustained by clusters [12], [13], clustering is associated with greater inequality and rich-poor segregation [10], and popularity-biased encounter lets visible success compound [11]. | Gini falls monotonically with giver fraction on every topology, as prior models predict. |
| **H3** | **Emergent segregation:** Strategy assortativity rises over time from its initial value without any homophily rule being coded. Givers retain ties to givers; takers are structurally quarantined, with lower realised degree and mean tie weight than givers. | Assortativity stays flat; takers keep giver-equivalent degree and tie weight. |
| **H4** | **The poverty trap is equal:** The worst city for a median citizen is not the most unequal one but the least prosperous. Taker-dominated cities converge to low wealth *and* low Gini. Median wealth, not Gini, best separates good from bad cities. | Taker-dominated cities show high Gini, or median wealth fails to separate outcomes that Gini does. |

H2 is the headline result. H1 and H3 describe the mechanisms expected to produce it; H4 is the
interpretation that follows if it holds. Because H2 runs against existing results, either
outcome is informative: confirming it identifies conditions under which cooperation stops
equalising a society, and refuting it shows the equalising effect survives realistic,
structured encounter.

## 3. Background

### 3.1 The Prisoner's Dilemma

- A two-player, single-round game, where each player independently chooses to cooperate or
  defect, with payoffs that typically follow in the matrix [14, Ch. 12]:

| | Opp C | Opp D |
|---|---|---|
| You C | 3 | 0 |
| You D | 5 | 1 |

- The typical payoff ordering is `T > R > P > S` (temptation > reward > punishment > sucker's
  payoff), with the additional constraint `2R > T + S` so that alternating exploitation is
  worse than sustained cooperation.
- The single-round Nash equilibrium is mutual defection, which is Pareto-inferior; this gap
  between individually rational and collectively optimal behaviour is the phenomenon the
  project is interested in.
- **This ordering is what makes the wealth story work:** `2R = 6 > T + S = 5 > 2P = 2`, so a
  mutually cooperative pair *creates* more total wealth than an exploitative pair, which
  creates more than a mutually defecting pair. The game is positive-sum, and the surplus a
  city generates follows directly from how its citizens pair up and behave.

### 3.2 The Iterated Prisoner's Dilemma

- The Prisoner's Dilemma game played repeatedly between the same two players, so that a
  player's current choice can be conditioned on the opponent's past behaviour.
- Repetition makes reciprocal strategies possible: cooperation can become viable given a
  long-term relationship [15].
- Typical strategies (each agent is assigned one at initialisation and keeps it for the whole
  run):

| Strategy | Behaviour | Nice | Retaliates | Forgives | Class | Role in our model |
|---|---|:--:|:--:|:--:|:--:|---|
| **Always Cooperate** (AllC) | Cooperates unconditionally. | Yes | No | n/a | Giver | Exploitable baseline; measures how much free-riding the city permits. |
| **Always Defect** (AllD) | Defects unconditionally. | No | n/a | No | Taker | The pure extractor; tests whether decay and rewiring can quarantine takers. |
| **Tit-for-Tat** (TFT) | Cooperates first, then copies opponent's previous move. | Yes | Yes | Yes | Giver | The Axelrod benchmark; our primary reciprocal reference point. |
| **Generous TFT** (GTFT) | TFT, but forgives a defection with probability *g*. | Yes | Yes | Strong | Giver | Breaks retaliation deadlocks; matters once noise is enabled. |
| **Tit-for-Two-Tats** (TF2T) | Defects only after two consecutive defections. | Yes | Yes | Strong | Giver | More noise-tolerant than TFT, but more exploitable. |
| **Grudger / Grim Trigger** | Cooperates until first defection, then defects forever. | Yes | Strong | No | Giver | Maximally unforgiving; under decay its dead ties should starve and be replaced. |
| **Pavlov** (Win-Stay-Lose-Shift) | Repeats last move if it scored well (T or R), switches otherwise. | Yes | Yes | Yes | Giver | Exploits AllC, recovers from noise; behaves differently to TFT under our weighting. |
| **Random** | Cooperates with probability 0.5. | No | No | n/a | Taker | Control; establishes a noise floor for interpreting results. |

- "Nice" means never the first to defect. Nice, retaliating and forgiving are among the
  properties Axelrod identified in strategies that did well in his tournaments
  [16], [14, Ch. 12].
- The **giver / taker** column is our coarse-graining for the primary experimental axis
  (Section 6): a city's *giver fraction* is the proportion of its agents drawn from the giver
  class. Within-class composition is held at a fixed default, so a single number
  characterises a city.
- Outcomes are sensitive to the environment: the composition of the strategy pool, the
  presence of noise (misimplemented or misread moves), and, the focus of this project,
  **who plays whom**.

### 3.3 Inspiration and Acknowledgements

- The inspiration for this project is the Veritasium video on the Prisoner's Dilemma and
  Axelrod's tournaments [17].
- The video's key takeaways are:
    - Reciprocity beats both pure exploitation and pure altruism.
    - Forgiveness matters once noise is introduced.
    - Evolutionary/population dynamics can shift when certain strategies dominate over time.
- The underlying game theory is attributed to the experimental games of Flood and Dresher
  [18] and to Axelrod and Hamilton [15]; the video is credited as
  the accessible presentation that motivated the topic, not as a primary source.

### 3.4 Relationship to *Think Complexity*

Two chapters of Downey's *Think Complexity* [14] are the closest textbook
precedents, and this project extends both.

**Chapter 12, Evolution of Cooperation:** Downey simulates a population playing the iterated
Prisoner's Dilemma. In each step every agent plays two opponents, randomly chosen when the
population is shuffled; strategies evolve through differential survival and mutation; and
outcomes are reported as mean fitness and the "niceness" of the population
[14, Ch. 12]. His own exercises point towards structure: he suggests that playing
the same neighbours repeatedly "might make it easier for a minority strategy to invade a
majority, by taking advantage of locality" [14, Ex. 12.1]. We take up that
suggestion directly, but change three things: encounter is shaped by a social graph rather
than shuffling, strategies are fixed so that composition is a controlled variable, and the
outcome is the *distribution* of payoffs rather than their mean.

**Chapter 9, Agent-Based Models:** Sugarscape produces heavy-tailed wealth inequality, but its
wealth is *harvested from an environment* by agents who differ in vision, metabolism and
lifespan. Downey calls the resulting distribution "a weak example" of emergence precisely
because it "could reasonably [be predicted] based on the distributions of vision, metabolism,
and lifespan" [14, Ch. 9]. In our model agents differ *only* in strategy, and wealth
is *created by interaction* between them, so any inequality that appears cannot be read off
agent attributes. It must come from who meets whom, which makes it a stronger test of
emergent inequality. The same chapter's Schelling model shows segregation emerging from mild
individual preferences [14, Ch. 9]; H3 asks whether segregation emerges with no
preference rule at all.

The candidate city topologies (Erdős–Rényi, Watts–Strogatz and Barabási–Albert graphs) are
the models of Chapters 2 to 4 [14, Chs. 2-4].

### 3.5 Related Literature

**Network reciprocity:** Cooperation can survive on a spatial lattice where it would die in a
well-mixed population, because cooperators form clusters that interact mostly with each other
[12]. Heterogeneous, scale-free networks were subsequently argued to promote
cooperation further [19]; the wider theory of evolutionary games on graphs is
reviewed in [20]. Human experiments temper this: degree heterogeneity does not
promote cooperation when people play the Prisoner's Dilemma [21]. Our model
inherits the theoretical assumption those experiments question, and this is treated as a
limitation.

**Dynamic networks and tie strength:** When people can rewire their ties, cooperation is
sustained in experiments [22]; clusters of cooperators emerge endogenously and insulate
conditional cooperators from exploitation, even without reputational information
[13]. Strong ties, operationalised as relationship duration, promote cooperation in
dynamic networks [23]. Tie-decay networks formalise the distinction our model
relies on, between *interactions* as discrete contacts and *ties* as relationship strengths
that decay over time [24].

**Distance and encounter:** In a large social network, the probability of befriending someone
is inversely proportional to the number of people closer to you, and about one-third of
friendships are independent of geography [25]. In geo-located Twitter data,
link lengths follow approximately `P(d) ~ 1/d` at larger scales [26]. Both describe
geographic distance; we borrow the functional form for graph path length, which is an
analogy rather than a calibration.

**Inequality in cooperation games:** Network structure, network fluidity and institutions
shape how earnings are distributed [10]; reputational information affects
dispersion differently in rewired and strategically updated networks [9];
incentives and network coupling can produce jointly high cooperation and low Gini, most
clearly on regular rather than scale-free networks [8]; wealth visibility amplifies
inequality [11]; and inequality in endowments and productivity changes whether
reciprocity can sustain cooperation at all [7].

### 3.6 Novel Contribution

1. **Distribution as the outcome, in a textbook setting that reports means:** Downey's
   Prisoner's Dilemma simulation reports mean fitness and niceness [14, Ch. 12]. We
   report the Lorenz curve and Gini coefficient of accumulated wealth alongside mean and median
   prosperity, and ask whether they move together.
2. **Encounter is a modelled mechanism, not an assumption:** Rather than shuffled pairing or
   playing every edge each round, partner choice is probabilistic and jointly driven by graph
   distance, tie strength and partner popularity, with the distance term's functional form
   taken from empirical studies of social ties [25], [26].
3. **Emergent inequality without heterogeneous endowments:** Unlike Sugarscape, agents do not
   differ in ability or environment [14, Ch. 9], and unlike experimental work that
   seeds inequality in endowments [11], [7], all agents start with equal
   wealth. Any inequality is produced by interaction structure.
4. **A direct test of a contested relationship:** H2 tests whether the negative
   cooperation-inequality relationship reported in network models [8] survives
   clustered, popularity-biased encounter, which experimental evidence associates with greater
   inequality [10].
5. **The composition:** Weighted ties with time decay, distance-decayed popularity-biased
   partner choice, and severance-and-replacement rewiring each have precedents
   [22], [13], [24], [25]; we have not found them
   combined, and not with wealth distribution as the measured outcome.

## 4. Model Overview

One simulation run is **one city**, observed over `T` rounds.

```
initialise city (N agents, topology, strategy mix, wealth = 0)
for each round t in 1..T:
    1. each agent selects interaction partners      (distance + weight + popularity)
    2. each selected pair plays one PD round        (strategy, per-partner history)
    3. payoffs credited to agent wealth             (scaled by resource density rho)
    4. tie weights reinforced on contact
    5. tie weights decayed by time since contact
    6. ties below threshold severed
    7. severed slots refilled with new partners
    8. popularity ratings recomputed
    9. metrics recorded
```

Every stochastic step draws from a single seeded `numpy.random.Generator` passed through the
model; there is no global RNG state and no unseeded call. A run is fully reproducible from
`(seed, config)`.

## 5. Model Components

### 5.1 The City Graph

- `N` agents as nodes, with `N` in the low hundreds: large enough for network-scale effects,
  small enough for a two-axis sweep with repeats.
- Initial topology from a configurable generator. **Erdős–Rényi** is the neutral default;
  **Watts–Strogatz** (small-world) and **Barabási–Albert** (scale-free) form the secondary
  experimental axis [14, Chs. 2-4]. Topology is expected to matter to H2 in
  particular, since the cooperation-inequality relationship has been reported to differ
  between regular and scale-free networks [8].
- Undirected. An edge means "these two know each other"; it does **not** mean they interact
  every round.
- Each agent is assigned a strategy from the table in Section 3.2, drawn from a configurable
  giver/taker mix, and keeps it for the run.
- Edge weights start at a small uniform `w0`, so no tie starts privileged. All agents start
  with zero wealth.

### 5.2 Wealth and the Resource Scale

- Payoffs are denominated in a per-capita **resource density** `rho`: the payoff matrix used
  is the Section 3.1 matrix multiplied by `rho`. A city's wealth-generating capacity therefore
  scales with its population without the pool being fixed.
- **Wealth is created, not redistributed:** Each interaction adds `rho x payoff` to each
  participant, so total city wealth at round `T` is a direct readout of how the city's
  citizens have treated each other.
- A conserved pool, from which interactions merely transfer wealth, is deliberately not used:
  conservation forces `2R = T + S = 2P`, which destroys the Prisoner's Dilemma payoff ordering
  and with it the phenomenon under study.
- An optional **subsistence cost** `c` is deducted per agent per round, so that wealth can fall
  as well as rise. Default `c = 0`; enabled as a robustness check.

### 5.3 Encounter: Who Plays Whom

Each round, every agent `i` initiates `B` interactions (default `B = 1`, reading a round as a
week with one meaningful encounter per person). Agent `i` chooses each partner `j` from its
candidate set with probability proportional to an **attractiveness score**:

```
s_ij          = (w_ij + eps)^beta  x  R_j^gamma  x  d_ij^(-lambda)
P(i picks j)  = s_ij / sum_k s_ik
```

- `d_ij` is the **graph path length** from `i` to `j`. The `d^(-lambda)` term is the distance
  decay, with default `lambda = 1` following the `P(d) ~ 1/d` relationship observed in social
  ties [26]. Candidates are restricted to within `d_max` hops (default 3), keeping the
  calculation cheap and the process local. At `d = 1` the term equals 1, so neighbours are
  unpenalised.
- `w_ij` is **tie strength**, the accumulated interaction history (Section 5.4): people are
  more likely to meet those they already meet often. `eps` is a small constant giving any
  candidate within `d_max` a non-zero probability, which is what allows new ties to form;
  its presence reflects the finding that a substantial share of real friendships do not follow
  the distance relationship [25].
- `R_j` is the **popularity rating** of `j` (Section 5.5).
- `beta`, `gamma` and `lambda` set the relative pull of habit, status and proximity. They are
  held at defaults outside the robustness checks.

An agent can also be *chosen* by others, so realised interaction counts vary between agents
even though initiated counts do not. **Received interactions are capped at `c_max`**: a
popular person has more encounters than an unpopular one, but not unboundedly more, because
attention is finite. The cap prevents the popularity term producing a single runaway winner.

A **rank-based** distance term, in which weight is inversely proportional to the number of
agents closer to `i` [25], is retained as a robustness variant. On small graphs,
where path lengths take only a few integer values, it produces coarse ties between candidates,
so it is not the default.

### 5.4 Ties: Reinforcement, Decay, Severance

The model separates **interactions**, which are discrete contacts, from **ties**, which carry
a relationship strength that decays over time, following the tie-decay formalism of
[24].

- **Reinforcement:** each interaction increments the tie's weight, `w <- w + delta`. Tie
  strength accumulating with sustained interaction reflects the role of strong ties in
  sustaining cooperation [23].
- **Decay:** weight decays with time since last contact, `w(t) = w x exp(-alpha x dt)`,
  applied lazily when the tie is evaluated. This models *neglect*, not punishment: an unused
  tie fades regardless of how its last interaction went.
- **Severance:** a tie whose weight falls below the threshold `theta` is removed.
- **Replacement:** the freed slot is refilled by drawing a new partner with the Section 5.3
  rule, most often a friend of a friend, because the distance term makes nearby candidates
  far more likely. Rewiring is therefore local and structure-preserving.

Severance is **absolute** rather than **comparative**. A comparative rule, in which an agent
swaps a tie for a better alternative, cannot operate here: a prospective tie has `w ~ 0` and
would never beat an incumbent, so the network would freeze. Under the absolute rule, takers
are still excluded, because they are chosen less, their ties go unused, and those ties decay
below `theta`.

Two decays operate, and they do different work: **distance decay** shapes *who you are likely
to meet*; **time decay** shapes *which relationships survive*. Encounter and persistence are
distinct social processes, and keeping them separate is what allows each to be varied and its
effect measured independently.

### 5.5 Popularity

`R_j` is a bounded, slowly updating rating of how much cooperative value `j` has recently
returned to its partners: a composite of weighted degree and behaviour, normalised to
`[R_min, R_max]` with `R_min > 0` so that no agent becomes unreachable.

It serves two roles in the hypotheses:

1. **Structural quarantine (H3):** A taker returns little value, so its rating falls, it is
   chosen less, and its ties decay from disuse and are severed. No punishment or ostracism
   rule is coded; exclusion is emergent.
2. **Concentration of surplus (H2):** Good partners are chosen more, up to the `c_max` cap,
   and so accumulate wealth faster than equally cooperative but less visible agents. This is
   the modelled analogue of wealth visibility amplifying inequality [11].

The rating is not omniscient reputation. It updates with a lag and is bounded, so agents
respond to a coarse social signal rather than to perfect information about each other.

### 5.6 Connection Style (Secondary)

Agents may carry a fixed **connection style**: broad-shallow (a high degree cap `k_broad`,
attention spread thinly) or narrow-deep (`k_narrow`, attention concentrated). Because
reinforcement is per interaction and decay is per unit time, equilibrium tie weight scales
roughly as `B / (k x alpha)`. Spreading attention more thinly lowers the steady-state weight
of every tie, and above a critical `alpha` broad-shallow ties should fail to sustain
themselves at all.

Connection style is a **robustness check, not an experimental axis.** The default population
is homogeneous in style; a mixed population is used to test whether H1-H4 survive
heterogeneity in how agents allocate attention.

## 6. Experimental Design

### 6.1 Progressive Model Build

The model is built in stages, each a named, runnable model with its own validation. Each
stage must reduce to its predecessor under null parameters, which is the main correctness
guarantee and allows any change in behaviour to be attributed to the mechanism just added.

| Stage | Adds | Validates against |
|---|---|---|
| **M0**: unstructured | Random pairing, no graph. | Known tournament behaviour: TFT competitive, AllD exploits AllC [16], [14, Ch. 12]. |
| **M1**: static graph | Fixed topology, every edge played each round. | Network reciprocity: cooperation sustained at mixes where M0 collapses [12]. |
| **M2**: weighted ties | Reinforcement, time decay, severance. | Tie weights reach a steady state; `alpha = 0` reduces to M1. |
| **M3**: encounter | Distance-decayed, popularity-biased partner choice. | `lambda = beta = gamma = 0` reduces to uniform choice over neighbours. |
| **M4**: rewiring | Severed-tie replacement. Full model. | Graph stays connected; turnover rate is bounded. |

### 6.2 Sweeps

- **Primary axis (giver fraction):** 0.0 to 1.0 in steps of 0.1. This is the "different
  mixes of agents" axis and answers the research question directly.
- **Secondary axis (topology):** Erdős–Rényi, Watts–Strogatz, Barabási–Albert. Tests whether
  network structure changes the H2 relationship.
- **Tertiary axis (decay rate `alpha`):** three levels (low, default, high), not a full sweep;
  enough to show results are not an artefact of one decay setting.
- **Repeats:** 20-30 seeds per cell. Every result is reported as a mean with a confidence
  interval, never from a single run.
- All other parameters (payoff values, `B`, `c_max`, `beta`, `gamma`, `lambda`, `theta`,
  `N`, `rho`) are held at documented defaults.

### 6.3 Parameter Calibration

Before the sweep, a calibration pass fixes defaults so that at a 50/50 mix neither extreme
outcome is forced: cooperation neither collapses to zero nor saturates at one, and tie
turnover is non-zero but not total. A sweep whose baseline is already pinned at an extreme
cannot reveal a threshold. Calibration runs and their outcomes are reported alongside the
main results.

## 7. Measures

### 7.1 Quantitative

**Prosperity**
- Mean and **median** wealth per capita at round `T`, and their trajectories. The gap between
  mean and median is itself evidence for or against H2.
- Total city wealth and wealth growth rate.

**Distribution**
- **Gini coefficient** of wealth over time (the primary H2 measure).
- **Lorenz curves**, one per giver fraction, overlaid.
- Wealth share of the top 10% and bottom 50%.
- Mean wealth **by strategy**, to identify who captures the surplus.

**Behaviour**
- Cooperation rate over time, overall and by strategy.
- Fraction of interactions that are C/C, C/D and D/D (the direct link from behaviour to
  surplus).

**Structure (in-groups and out-groups)**
- **Strategy assortativity** over time (the H3 measure).
- **Modularity** of the final weighted graph under community detection, and the **strategy
  purity** of detected communities: are emergent clusters giver clusters?
- Realised degree and mean tie weight, givers versus takers (the H3 quarantine measure).
- Tie turnover rate, edge count, and the distribution of tie weights.

### 7.2 Qualitative

- **Network visualisations** at several time points: nodes coloured by strategy and sized by
  wealth, edges weighted by tie strength. Segregation and concentration of wealth should be
  directly visible.
- An **animation** of one representative run.
- **Three city case studies** (giver-dominated, 50/50 and taker-dominated), each described
  as a narrative of what happened, to whom, and why.
- **Comparison with the empirical record:** whether the model reproduces context-dependent
  returns to cooperation [2], whether bridging ties matter more than cohesion
  [3], and whether it reproduces or contradicts the negative cooperation-inequality
  relationship [8]. A contradiction is a result, provided the responsible model
  assumption can be identified.

### 7.3 Headline Figure

Two panels sharing an x-axis of giver fraction: **mean and median wealth** above, **Gini**
below, one line per topology, with confidence bands. If H2 holds, the panels diverge visibly,
with wealth rising while Gini fails to fall.

## 8. Validation

- **Reduction tests:** each stage collapses to its predecessor under null parameters
  (Section 6.1). Automated.
- **Baseline reproduction:** M0 reproduces known unstructured IPD behaviour
  [14, Ch. 12] before any novel mechanism is switched on.
- **Degenerate cases:** an all-AllC city should show near-maximal wealth and low Gini; an
  all-AllD city near-minimal wealth and low Gini; identical agents on a regular graph should
  show Gini near zero. A failure indicates an accounting error.
- **Seed stability:** results are averaged over repeats and their spread shown. A conclusion
  that depends on a single seed is not reported as a conclusion.

## 9. Scope

The model is implemented in Python 3 with `networkx`, `numpy` and `matplotlib`, and presented
as Jupyter notebooks backed by importable model code.

### Core

- Fixed strategy per agent, assigned at initialisation.
- Positive-sum wealth accumulation scaled by resource density `rho`.
- Distance-decayed, popularity-biased partner choice with a received-interaction cap.
- Weighted ties with time decay, absolute severance and local replacement.
- Progressive build M0-M4 with reduction tests.
- Giver fraction x topology sweep, three decay levels, 20-30 seeds per cell.
- Full prosperity, distribution, behaviour and structure metric suite, and the headline figure.

### Stretch targets

- Connection style (Section 5.6) as a robustness check.
- Subsistence cost `c > 0`.
- Rank-based distance term as a robustness variant.
- Action noise.
- Animation of a representative run.

### Out of scope

- Strategy evolution, mutation or imitation; agents never change disposition.
- Migration between cities; every run is a single, closed city.
- A full combinatorial sweep over payoff values, `B`, `c_max` and the encounter exponents.

## References

[1] C. Bjørnskov, "Social trust and patterns of growth," *Southern Economic Journal*, vol. 89, no. 1, pp. 216-237, Jul. 2022, doi: 10.1002/soej.12590.

[2] X. Xue, W. R. Reed, and R. C. M. van Aert, "Social capital and economic growth: A meta-analysis," *Journal of Economic Surveys*, vol. 39, pp. 1395-1432, Sep. 2025, doi: 10.1111/joes.12660.

[3] R. Chetty et al., "Social capital I: Measurement and associations with economic mobility," *Nature*, vol. 608, no. 7921, pp. 108-121, Aug. 2022, doi: 10.1038/s41586-022-04996-4.

[4] G. Barone and S. Mocetti, "Inequality and trust: New evidence from panel data," *Economic Inquiry*, vol. 54, no. 2, pp. 794-809, Apr. 2016, doi: 10.1111/ecin.12309.

[5] J. Graafland and B. Lous, "Income inequality, life satisfaction inequality and trust: A cross country panel analysis," *Journal of Happiness Studies*, vol. 20, no. 6, pp. 1717-1737, Aug. 2019, doi: 10.1007/s10902-018-0021-0.

[6] G. Kanitsar, "The inequality-trust nexus revisited: At what level of aggregation does income inequality matter for social trust?," *Social Indicators Research*, vol. 163, no. 1, pp. 171-195, Aug. 2022, doi: 10.1007/s11205-022-02894-w.

[7] O. P. Hauser, C. Hilbe, K. Chatterjee, and M. A. Nowak, "Social dilemmas among unequals," *Nature*, vol. 572, no. 7770, pp. 524-527, Aug. 2019, doi: 10.1038/s41586-019-1488-5.

[8] L.-L. Jiang, Y.-M. Li, W.-J. Li, and A. Szolnoki, "Combined effect of incentives and coupling in multigames in two-layer networks," *Journal of Physics: Complexity*, vol. 6, no. 1, Art. no. 015003, Mar. 2025, doi: 10.1088/2632-072X/ada844.

[9] M. Tsvetkova, "The effects of reputation on inequality in network cooperation games," *Philosophical Transactions of the Royal Society B*, vol. 376, no. 1838, Art. no. 20200299, Nov. 2021, doi: 10.1098/rstb.2020.0299.

[10] M. Tsvetkova, C. Wagner, and A. Mao, "The emergence of inequality in social groups: Network structure and institutions affect the distribution of earnings in cooperation games," *PLOS ONE*, vol. 13, no. 7, Art. no. e0200965, Jul. 2018, doi: 10.1371/journal.pone.0200965.

[11] A. Nishi, H. Shirado, D. G. Rand, and N. A. Christakis, "Inequality and visibility of wealth in experimental social networks," *Nature*, vol. 526, no. 7573, pp. 426-429, Oct. 2015, doi: 10.1038/nature15392.

[12] M. A. Nowak and R. M. May, "Evolutionary games and spatial chaos," *Nature*, vol. 359, no. 6398, pp. 826-829, Oct. 1992, doi: 10.1038/359826a0.

[13] D. Melamed, A. Harrell, and B. Simpson, "Cooperation, clustering, and assortative mixing in dynamic networks," *Proceedings of the National Academy of Sciences*, vol. 115, no. 5, pp. 951-956, Jan. 2018, doi: 10.1073/pnas.1715357115.

[14] A. B. Downey, *Think Complexity: Complexity Science and Computational Modeling*, 2nd ed. Sebastopol, CA, USA: O'Reilly Media, 2018.

[15] R. Axelrod and W. D. Hamilton, "The evolution of cooperation," *Science*, vol. 211, no. 4489, pp. 1390-1396, Mar. 1981, doi: 10.1126/science.7466396.

[16] R. Axelrod, *The Evolution of Cooperation*. New York, NY, USA: Basic Books, 1984.

[17] Veritasium, "This game theory problem will change the way you see the world," YouTube, Dec. 23, 2023. [Online]. Available: https://www.youtube.com/watch?v=mScpHTIi-kM (accessed Sep. 16, 2026).

[18] M. M. Flood, "Some experimental games," *Management Science*, vol. 5, no. 1, pp. 5-26, Oct. 1958, doi: 10.1287/mnsc.5.1.5.

[19] F. C. Santos and J. M. Pacheco, "Scale-free networks provide a unifying framework for the emergence of cooperation," *Physical Review Letters*, vol. 95, no. 9, Art. no. 098104, Aug. 2005, doi: 10.1103/PhysRevLett.95.098104.

[20] G. Szabó and G. Fáth, "Evolutionary games on graphs," *Physics Reports*, vol. 446, no. 4-6, pp. 97-216, Jul. 2007, doi: 10.1016/j.physrep.2007.04.004.

[21] C. Gracia-Lázaro et al., "Heterogeneous networks do not promote cooperation when humans play a Prisoner's Dilemma," *Proceedings of the National Academy of Sciences*, vol. 109, no. 32, pp. 12922-12926, Aug. 2012, doi: 10.1073/pnas.1206681109.

[22] D. G. Rand, S. Arbesman, and N. A. Christakis, "Dynamic social networks promote cooperation in experiments with humans," *Proceedings of the National Academy of Sciences*, vol. 108, no. 48, pp. 19193-19198, Nov. 2011, doi: 10.1073/pnas.1108243108.

[23] D. Melamed and B. Simpson, "Strong ties promote the evolution of cooperation in dynamic networks," *Social Networks*, vol. 45, pp. 32-44, Mar. 2016, doi: 10.1016/j.socnet.2015.11.001.

[24] W. Ahmad, M. A. Porter, and M. Beguerisse-Díaz, "Tie-decay networks in continuous time and eigenvector-based centralities," *IEEE Transactions on Network Science and Engineering*, vol. 8, no. 2, pp. 1759-1771, Apr. 2021, doi: 10.1109/TNSE.2021.3071429.

[25] D. Liben-Nowell, J. Novak, R. Kumar, P. Raghavan, and A. Tomkins, "Geographic routing in social networks," *Proceedings of the National Academy of Sciences*, vol. 102, no. 33, pp. 11623-11628, Aug. 2005, doi: 10.1073/pnas.0503018102.

[26] J. Szüle, D. Kondor, L. Dobos, I. Csabai, and G. Vattay, "Lost in the city: Revisiting Milgram's experiment in the age of social networks," *PLOS ONE*, vol. 9, no. 11, Art. no. e111973, Nov. 2014, doi: 10.1371/journal.pone.0111973.
