# Bayesian opinion dynamics with cash transfers

A runnable first experiment following the discrete belief dynamics in
[YutongB/Opinion-Dynamics](https://github.com/YutongB/Opinion-Dynamics), with
payment-weighted social influence and conserved cash. The agreed first model has
allies only, shared coin-toss evidence, and one randomly chosen neighbour per
payer per round. Payments are a fixed amount, a fraction of the payer's current
wealth, or the maximum of the two (the hybrid rule). The recipient gains the
money and moves toward the payer's belief.

This is an exploratory agent-based model. Bayesian conditioning applies to the
coin observation; social pooling is an assumed response to payment. It is not a
model of fully rational Bayesian inference from other agents' correlated reports,
nor a general equilibrium economy.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m bayes_cash --output outputs/demo
python -m pytest -q
```

The command runs **both uniform and loguniform initial wealth**, each with fixed,
fractional, hybrid, and no-payment scenarios. The latter is a control for learning from
public evidence. A nonempty output directory is refused to protect previous runs.

For another paired realization:

```bash
python -m bayes_cash --seed 43 --rounds 1000 --output outputs/seed43
```

All model parameters are exposed by `python -m bayes_cash --help`. For example:

```bash
python -m bayes_cash --fixed-payment 2 --wealth-fraction 0.1 \
  --wealth-min 1 --wealth-max 100 --output outputs/larger-payments
```

`--no-public-evidence` is an optional social-only diagnostic. The default and the
agreed experiment retain shared coin evidence.

## Model and ordering

At time t, agent i has cash W_i and a probability mass function p_i(k) on a common
grid theta_k in [0,1]. Defaults are 100 agents, a connected Barabasi-Albert network
with attachment parameter 3, 500 rounds, 21 grid points, true coin bias 0.6, and
social rate mu = 0.25. All agents are persuadable allies; there are no partisans.
NetworkX builds the graph; NumPy implements the equations, without requiring the
upstream package's compiled graph-tool dependency.

### 1. Initial state

For each agent, independently draw a Gaussian location a_i uniformly on [0,1]
and width s_i uniformly on [0.2,0.8]. These match the upstream prior conventions:

```math
p_i(0,k)=\frac{\exp[-(\theta_k-a_i)^2/(2s_i^2)]}
                   {\sum_\ell\exp[-(\theta_\ell-a_i)^2/(2s_i^2)]}.
```

These are discrete masses normalized by a sum, not continuous density values to
integrate with a quadrature rule. A Gaussian's input location is generally not
its mean after restricting and normalizing to [0,1]. Diagnostics use the actual
mass-function mean.

For u_i drawn independently and uniformly on [0,1], wealth is either

```math
W_i(0)=W_{\min}+(W_{\max}-W_{\min})u_i
\quad\text{or}\quad
W_i(0)=\exp[\log W_{\min}+u_i\log(W_{\max}/W_{\min})].
```

Defaults are W_min = 1 and W_max = 100, in arbitrary cash units. Wealth is drawn
independently of beliefs and network position. There is **no rescaling of the
realized total wealth**. The two distributions use the same underlying u_i,
preserving wealth rank for a controlled comparison, but have different total
and typical wealth. Their payment modes should first be compared within each
distribution.

### 2. Shared Bayesian evidence

All agents observe the same coin toss y_t, independently drawn across rounds
from Bernoulli(theta_true). For heads L_k = theta_k; for tails L_k = 1-theta_k:

```math
q_i(k)=\frac{L_k(y_t)p_i(k)}{\sum_\ell L_\ell(y_t)p_i(\ell)}.
```

An observation with zero predictive probability raises an error. There is no
probability floor or silent replacement of an impossible posterior. In the
social-only diagnostic, q_i = p_i.

### 3. Directed payments, using opening wealth

Each payer j chooses one neighbour r_j uniformly, independently of opinions,
cash, and affordability. Choices for **all rounds** are generated before any
scenario is run, so failed payments cannot alter later random contacts.

```math
m_j=\begin{cases}
c\,\mathbf{1}_{W_j\geq c},&\text{fixed payment},\\
fW_j,&\text{fractional payment},\\
\max(c,fW_j)\,\mathbf{1}_{W_j\geq c},&\text{hybrid payment},
\end{cases}
\qquad P_{ji}=m_j\mathbf{1}_{r_j=i}.
```

Defaults are c = 1 and f = 0.05. The fraction is of the **payer's opening wealth**,
not the recipient's wealth or the payer's initial wealth. Fixed payments that
cannot be funded are skipped entirely; no partial payments, debt, clipping, or
credit are introduced. Incoming money becomes spendable in the next round.

The hybrid rule requests max(c, f W_j), and pays only if the opening balance
covers the whole request. Because 0 <= f <= 1, this is equivalent to requiring
W_j >= c. With c=1 and f=0.05, agents below 1 skip, agents from 1 to 20 pay 1,
and agents above 20 pay 5% of their wealth. An agent at W_j=c can spend its entire
balance: c is an eligibility threshold, not a balance that must remain afterward.
For f=0 the hybrid reduces to fixed payments; for c=0 it reduces to fractional
payments.

After all payments have been computed, cash settles simultaneously:

```math
M_i=\sum_j P_{ji},\qquad W_i'=W_i-m_i+M_i.
```

Every payment is debited from its payer and credited to its recipient, so total
cash is conserved. Simultaneous settlement prevents agent iteration order from
changing who can afford to pay.

### 4. Payment-weighted social pooling

Let d_i be agent i's undirected network degree and m_ref > 0 a reference payment
(default 1). Retain a cash-valued self-weight

```math
K_i=d_i m_{\rm ref}\frac{1-\mu}{\mu},\qquad 0<\mu<1.
```

All recipients update synchronously using the same just-Bayesian-updated q:

```math
p_i'(k)=\frac{K_iq_i(k)+\sum_j P_{ji}q_j(k)}{K_i+M_i}.
```

For M_i > 0, this is a convex mixture of q_i and the payment-weighted payer
distribution, with total social weight

```math
\alpha_i=\frac{M_i}{K_i+M_i},\qquad
\frac{\partial\alpha_i}{\partial M_i}=\frac{K_i}{(K_i+M_i)^2}>0.
```

Thus an unpaid agent receives no social influence, a larger incoming amount has
a stronger bounded effect, and no negative-probability truncation is needed.
With several payers, their relative contributions are proportional to payment.
Belief updates use q, not neighbours' already socially updated beliefs.

**Calibration to the source:** if every neighbour pays m_ref, then M_i=d_i m_ref,
and the equation reduces exactly to

```math
p_i'=(1-\mu)q_i+\frac{\mu}{d_i}\sum_{j\sim i}q_j,
```

the upstream unweighted allies-only social step. This calibration is a limiting
check; our one-recipient-per-payer contact process differs from the upstream
all-neighbours-per-round process. In particular, high-degree recipients retain
more self-weight. A single payment m_ref gives alpha_i = mu/[d_i(1-mu)+mu], not
mu. This degree dependence is an explicit modelling choice.

## Saved data and diagnostics

Each run saves:

- `summary.json`: parameters, installed package versions, upstream revision,
  conventions, final diagnostics, and numerical conservation errors.
- `{loguniform,uniform}_trajectories.npz`: grid, network, initial distributions,
  initial wealth, shared evidence, recipient schedule, and full belief/wealth/
  payment histories for all four scenarios.
- `{loguniform,uniform}_metrics.csv`: opinion mean, disagreement, uncertainty,
  mean squared belief error, wealth summaries, Gini, total payments, and the
  fraction of agents who paid in each round.
- `{loguniform,uniform}_comparison.png` and `.pdf`: six time-series diagnostics.
- `wealth_redistribution.png` and `.pdf`: initial versus final wealth, with
  network degree indicated by colour.

For a scenario, beliefs have shape (T+1,N,K) and wealth (T+1,N), including the
initial state at index 0. Payments and recipients have shape (T,N). Transaction
`payments[t,j]` moves from j to `recipients[t,j]` and produces state t+1. This is
the complete transaction ledger, including attempted contacts with zero payment.
No pickle is needed to read the archives:

```python
import numpy as np

with np.load("outputs/demo/loguniform_trajectories.npz") as data:
    wealth = data["fractional_wealth"]
    payments = data["fractional_payments"]
    recipients = data["recipients"]
    mean_opinions = data["fractional_beliefs"] @ data["theta"]
```

Diagnostic belief MSE is the population average of
sum_k p_i(k)(theta_k-theta_true)^2; it includes both squared mean error and
within-agent uncertainty. Disagreement is the standard deviation of agent means,
which is a different quantity. All figures are exploratory diagnostics.

### Included demonstration

`outputs/demo` contains the default 100-agent, 500-round run with seed 42. Its
wealth Gini coefficients are:

| Initial wealth distribution | Initial | Final, fixed c=1 | Final, fraction f=0.05 | Final, hybrid |
| --- | ---: | ---: | ---: | ---: |
| Loguniform | 0.5594 | 0.9285 | 0.3945 | 0.6059 |
| Uniform | 0.3136 | 0.9001 | 0.3945 | 0.4087 |

All three paid scenarios reduce disagreement faster than the no-payment control.
Final population means are near 0.566, while the realized 500-toss heads fraction
is 0.568 (generating bias 0.6). Shared evidence dominates the mean opinion in this
realization. Maximum absolute total-cash drift across the runs is below 5e-12
cash units. Tests protect model and export contracts; the demonstration's
saved cash ledger and sampled belief updates were also independently replayed.
These observations are illustrative, not an ensemble result or convergence test.

## Why the payment rule and network both matter

Let g(W_j) be the actual payment after applying the affordability rule. Agent j
selects each of its d_j neighbours with probability 1/d_j, so its expected
payment to i is A_ij g(W_j)/d_j. Subtracting i's own outflow gives the exact
conditional drift:

```math
\mathbb E[W_i'-W_i\mid\boldsymbol W]
=\sum_j\frac{A_{ij}}{d_j}g(W_j)-g(W_i).
```

For fixed payments, when everyone is solvent this becomes
c (sum_j A_ij/d_j - 1). Receiving opportunities depend on the degrees of one's
neighbours as well as one's own degree. Hubs often receive more than they pay,
but becoming richer never increases their fixed outflow. Agents below c stop
paying temporarily; they can still receive money and resume paying later.

Even on a d-regular network without hubs, fixed payments have no wealth-dependent
restoring force while everyone is solvent. Each recipient gets c times a
Binomial(d,1/d) number of payments, so

```math
\mathbb E[\Delta W_i\mid\boldsymbol W]=0,\qquad
\operatorname{Var}(\Delta W_i\mid\boldsymbol W)=c^2(1-1/d).
```

Expanding (W_i+Delta W_i)^2 and summing gives an expected increase
N c^2(1-1/d) in sum_i W_i^2 for that round. Thus random transfers can broaden
wealth even without hubs. This expression assumes all agents can pay; it does
not imply that Gini must increase every round after agents become insolvent.

With fractional payments, the corresponding drift is

```math
\mathbb E[\Delta W_i\mid\boldsymbol W]
=f\left(\sum_j\frac{A_{ij}W_j}{d_j}-W_i\right).
```

On a regular network this restores wealth toward neighbouring balances: rich
agents pay more. More generally, if S is conserved total wealth, the exact
ensemble-mean recursion has stationary mean

```math
\overline W_i=\frac{S d_i}{\sum_j d_j},
```

since substituting this expression into the drift makes it zero. For a connected
graph and 0<f<1, the mean recursion converges to this profile. This is a statement
about **mean wealth**, not equality of every realized wealth trajectory or a
formula for mean Gini; random transfers retain fluctuations. At f=1, periodicity
can prevent convergence on bipartite graphs.

The hybrid rule adds the fractional outflow above c/f, limiting the fixed rule's
retention of wealth by rich agents. Below c/f it still charges solvent agents
more than the pure fractional rule, and below c it prevents payment. This can
leave greater inequality than pure fractional payments, particularly when many
initial balances lie below c/f. There is no general ordering theorem for its Gini.

### Paired network controls

Run the cash subsystem, which is exactly independent of beliefs in this model:

```bash
python scripts/check_network_effects.py --seeds 20 --rounds 5000 \
  --output outputs/network_controls
```

The included results use seeds 42--61, the same initial wealth for each seed
across all networks and rules, and identical contact schedules across rules
within each network. Networks are the original BA construction, a connected
degree-6 random regular graph, and a complete graph. All have 100 agents.
The first 500 BA rounds reproduce the main experiment exactly; independent seeded
continuation draws extend each trajectory to round 5000. This explicit prefix
preservation matters because the main generator draws contacts payer by payer:
calling it separately with a different total number of rounds would alter early
contacts for later payers.

Mean Gini at round 500, with sample standard deviation across the 20 seeds:

| Initial wealth | Network | Fixed | Fractional | Hybrid |
| --- | --- | ---: | ---: | ---: |
| Loguniform | BA | 0.926 +/- 0.014 | 0.369 +/- 0.016 | 0.580 +/- 0.033 |
| Loguniform | Regular | 0.505 +/- 0.028 | 0.088 +/- 0.006 | 0.137 +/- 0.034 |
| Loguniform | Complete | 0.504 +/- 0.019 | 0.089 +/- 0.006 | 0.136 +/- 0.031 |
| Uniform | BA | 0.900 +/- 0.019 | 0.369 +/- 0.016 | 0.385 +/- 0.017 |
| Uniform | Regular | 0.349 +/- 0.019 | 0.088 +/- 0.006 | 0.088 +/- 0.006 |
| Uniform | Complete | 0.351 +/- 0.017 | 0.089 +/- 0.006 | 0.089 +/- 0.006 |

Fixed payments retain much more inequality than fractional payments even without
hubs. The heterogeneous BA network strongly amplifies concentration. Extending
to 5000 rounds preserves that distinction: mean fixed-payment Gini is about
0.96 on BA, compared with 0.45--0.50 on the networks without hubs. These are
finite-time robustness checks, not established stationary limits. The BA network
has mean degree 5.82 versus 6 for the regular control; the complete network also
changes density and mixing. The comparison therefore does not assign an exact
causal percentage to degree heterogeneity alone.

The hybrid remains close to fractional payments for uniform initial wealth,
but separates more for loguniform initial wealth. The loguniform draws have
lower typical and total wealth with these same bounds, so many more balances
are affected by the fixed floor. This is a joint scale-and-shape comparison,
not an isolated effect of the distribution's shape.

`outputs/network_controls` saves per-seed outcomes in `per_seed.csv`, group
statistics and paired contrasts in `summary.json`, network/wealth snapshots and
Gini histories in `network_controls.npz`, and a PNG/PDF comparison figure.
Error bars show sample SD, not confidence intervals. Tests verify the cash-only
calculation against the full belief simulation for all three modes and networks.

## Interpretation limits and next decisions

- This first version has **wealth-to-opinion coupling**. Spending and recipient
  choice do not depend on beliefs, so changing beliefs cannot change wealth.
  The cash-transfer dynamics are autonomous, even though cash received finances
  later persuasion. There are no wages, production, goods, or utility optimisation.
- Common public evidence may overwhelm transient paid influence. Compare the
  no-payment control and early-time trajectories before interpreting consensus.
- The three payment rules do not have equal realized spending. For the
  same wealth bounds, uniform and loguniform initialization also imply different
  relative payment burdens. The outputs report actual spending and initial cash.
- Network position matters for redistribution as well as persuasion. Random
  neighbour selection can favour hubs; degree-coloured wealth plots help expose
  this. A wealth association alone is not evidence of persuasive success.
- A single seed is an illustration, not a statistical comparison. Repeat paired
  seeds and examine sensitivity to the wealth range, spending rate, social rate,
  network, and grid before making scientific claims. Tests establish software
  and conservation contracts, not empirical or model validity.
- For the next extension, the user proposed that payment **underweight opponents'
  repulsive influence**. Opponents and the functional form of that attenuation
  are intentionally deferred until after this allies-only experiment.

## Provenance

The source inspected was commit
[`be335d44c009add15388dbdc28b9fb2c42c5fb9e`](https://github.com/YutongB/Opinion-Dynamics/tree/be335d44c009add15388dbdc28b9fb2c42c5fb9e).
The relevant implementations are
[`step_simulation`](https://github.com/YutongB/Opinion-Dynamics/blob/be335d44c009add15388dbdc28b9fb2c42c5fb9e/src/lmbh/simulation/sim.py#L244),
[`UniformRandomPriorParams`](https://github.com/YutongB/Opinion-Dynamics/blob/be335d44c009add15388dbdc28b9fb2c42c5fb9e/src/lmbh/simulation/params.py#L120),
and the prior initialization in `simulation/initsim.py`. This package implements
the equations directly rather than vendoring upstream code. It reproduces the
allies-only equation and prior conventions, not the entire upstream feature set
or graph generator's exact random draws.

The underlying reference is Low and Melatos, *Discerning media bias within a
network of political allies and opponents: the idealized example of a biased
coin*, [arXiv:2112.10160](https://arxiv.org/abs/2112.10160),
[doi:10.1016/j.physa.2021.126722](https://doi.org/10.1016/j.physa.2021.126722).
The cash transfers, bounded cash weights, and payment schedule are extensions
specified for this project, not mechanisms attributed to that paper.
