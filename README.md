# Bayesian opinion dynamics with cash transfers

A runnable experiment following the discrete belief dynamics in
[YutongB/Opinion-Dynamics](https://github.com/YutongB/Opinion-Dynamics), with
payment-weighted social influence and cash transfers. Optional income adds an
explicit source of new money. Agents observe shared
coin-toss evidence and contact at most one neighbour per round. Agents can prefer
to give to allies or opponents, and recipients can require regular payments to
remain allies. Payments
are a fixed amount, a fraction of the payer's current wealth, or the maximum of
the two (the hybrid rule). The recipient gains the money. Allied payments buy
attraction; opponent payments weaken repulsion. The original allies-only model
remains the default and is preserved in commit `130cda2` and `outputs/demo`.

This is an exploratory agent-based model. Bayesian conditioning applies to the
coin observation; social pooling is an assumed response to payment. It is not a
model of fully rational Bayesian inference from other agents' correlated reports,
nor a general equilibrium economy.

## Run

### Interactive belief explorer

Start the local browser app with the same environment as the experiments:

```bash
.venv/bin/python -m bayes_cash.explorer
```

With uv, the committed lockfile pins the dependency versions:

```bash
uv run --locked python -m bayes_cash.explorer
uv run --locked python -m bayes_cash.explorer --help
```

Use `--port 8766` if the default port is already occupied, or `--port 0` to
choose an available port. Stop the local server with Ctrl-C.

Open `http://127.0.0.1:8765/`. The app calls the existing Python simulator and
needs no JavaScript dependencies or external services. Choose any subset of the
BA, connected random-regular (degree `2m`), and complete networks, then run the
experiment. Each run includes fixed, fractional, hybrid, and no-payment arms.
Controls expose public evidence, opponent probability, giver preferences,
payment-dependent loyalty, attenuation/tanh influence, proportional income and
its spending budget, wealth distribution, and model scales.
Expand **What the plots mean & model conventions** for a grouped reference to
every model and intervention parameter, including its units, allowed values,
timing, and interactions with the other settings.

Play, pause, step, or scrub the shared timeline. Select a network node or an agent
card to inspect its **discrete belief PMF**, compare payment rules, and follow its
history heatmap. Network glyphs and all-agent mini-plots show probability masses
on a fixed 0–1 vertical scale. The topology comparison holds the agent and payment
rule fixed. Edges show **current** connections and distinguish mutual allies,
mutual opponents, and mixed directed stances. The selected agent's relationship
table shows both directions; under the tanh law an opponent contact can still
carry attractive effective influence. Node positions stay fixed during playback.

The **Baseline · no interventions** and **Intervened** networks play side by side,
with the same layout, selected agent, payment rule, and displayed round. Selecting
a node in either view pauses both. The baseline runs the same model with an empty
intervention log, using the **exact same prepared inputs**: initial graph and
relationship signs, beliefs, cash, agent traits, evidence, scheduled recipients,
and contact uniforms. It is recomputed from those saved inputs, without reseeding
or generating a new network. With no logged interventions, both views are identical.

The selected-agent comparison shows both belief PMFs and baseline/intervened
values for beliefs, cash, and population metrics. Differences are **intervened
minus baseline**. Before the first effective intervention round the trajectories
match; after it, realized contacts and relationships may diverge even though the
underlying random inputs remain paired. The baseline uses the selected payment
rule, including its configured income and loyalty; choosing **No payment** is a
separate comparison. These differences describe this paired realization, not an
average intervention effect across random seeds.

Click a node to pause playback and choose an intervention from the grouped menu:

- **Behavior:** giving preference (uniform, allies, or opponents) and
  payment-dependent loyalty.
- **Money:** add or withdraw a specified amount of external cash once. This
  changes total money; it is not a transfer from another agent. A withdrawal
  exceeding the selected agent's opening cash in any scenario rejects the
  intervention and names the failing network, payment rule, and round.
- **Beliefs:** blend once toward a Gaussian target using center, width, and
  strength. On the existing theta grid, normalize the Gaussian by a sum, then
  apply `p_new = (1 - strength) * p_current + strength * target`. Strength 0
  leaves beliefs unchanged; strength 1 replaces them. Center and width describe
  the input Gaussian, not its discrete mean and standard deviation after
  normalization on the bounded grid.
- **Connections:** set an ally connection, set an opponent connection, or remove
  a connection to another agent. Setting a connection creates a missing edge or
  updates an existing one; both directed signs are assigned and that pair's
  unpaid clocks reset. Removing an absent edge is a no-op. Unrelated clocks and
  relationships are preserved. Isolated agents still observe public evidence
  and receive configured income, but have no contacts or social influence.

Choose **Effective from round** and apply the intervention. Round 51 means the change
is applied before update 51, preserving states 0–50. The default is the next
round after the displayed state; at the final state, choose an earlier round.
Future changes can be scheduled, and choosing an earlier round recomputes the
affected history. Behavior settings persist until superseded; cash and belief
changes happen once, with their consequences carried forward. They apply to the same
agent in every selected network and payment rule, preserving the paired
comparison. The initial conditions and underlying random draws stay fixed;
actual contacts can diverge when the edited behavior changes recipient choices.

The intervention log records the agent, effective round, and changed values.
Events execute before the round's evidence, contacts, income, and payments, with
log order breaking ties. Trait and connection assignments use the last assigned
value; cash changes accumulate and belief blends compose in order. Enabling
payment-dependent loyalty starts fresh unpaid-round clocks before that update.
Disabling it preserves current directed relationships. Re-enabling starts fresh
clocks again, while setting an already enabled trait to true preserves its clocks.
Payment amounts, preference probability, and loyalty duration remain global.
Running a new experiment from the initialization controls clears the old log.

Connection edits change the current degree used in each agent's self-weight.
Uniform givers retain the original prepared recipient when their neighbor set
matches its initial state; otherwise the stored contact uniform chooses among
sorted current neighbors. Preferential givers use the same stored uniforms with
their current relationship groups. No new random draws are made during editing.
Network-family labels describe the initial graph; edited graphs can differ.

External cash adjustments have a separate ledger, distinct from payments and
proportional income. Income uses opening wealth **after** the cash intervention,
so each update obeys `change in total cash = external adjustment + income`.
The viewer reports total cash, external adjustments, and their cumulative sum.
Withdrawing all cash is allowed; the Gini coefficient is then undefined and is
saved as JSON `null`, rather than assigned an arbitrary inequality value.

**Save experiment JSON** records the complete configuration, generated inputs
(including every network, initial belief and cash value, evidence, and contact
random draw), ordered interventions, and model/dependency identity.
**Load experiment JSON** in the local app reruns those exact inputs and edits,
and derives the matching baseline with an empty log.
Files from a different model source or Python/NumPy/NetworkX version are rejected
explicitly: use the recorded implementation to reproduce them. Reproducibility
is checked by exact trajectory equality in the current environment; this is not
a promise of bitwise equality across all hardware. Save the JSON before closing
the page if you want to keep the experiment; there is no automatic disk save.

**Save replay HTML** downloads the current experiment as a self-contained offline
replay, including both comparison trajectories, its intervention log, and experiment
JSON download. Playback,
node selection, and comparisons remain interactive; new interventions and model
settings require the local Python app. Alternatively, generate a replay from the
command line (existing files are never overwritten):

```bash
.venv/bin/python -m bayes_cash.explorer --export outputs/belief-replay.html
# Optional request JSON: {"config": {"rounds": 200}, "distribution": "uniform",
#                         "topologies": ["barabasi_albert", "regular"]}
.venv/bin/python -m bayes_cash.explorer --export outputs/custom-replay.html --config request.json
# A saved experiment JSON can also be passed to --config to reproduce its run.
```

The repository includes `outputs/belief-observatory.html` as a ready-to-open
offline demonstration and `outputs/belief-observatory-preview.jpg` as a design
snapshot. The editable interface lives in `src/bayes_cash/web/explorer.html`.

Viewer defaults use 24 agents, 160 rounds, and opponent probability 0.2 for a
responsive preview. The batch model defaults remain unchanged. Limits are
4–100 agents, 1–1000 rounds,
5–51 grid points, and four million exported belief, relationship, and cash-adjustment
values combined. Both trajectories count toward this limit when interventions
are logged; an empty-log run stores one trajectory for the two identical views.
Relationship histories count toward this limit when payment-dependent loyalty
or connection interventions are enabled in the corresponding trajectory.
Normally up to 501 display states include the initial and final states;
intervention boundaries additionally retain the states immediately before and
after each edit. All model rounds are computed.
The app shows sampled-frame status and steps through actual saved states without
interpolating. Request fewer agents, grid points, rounds, or networks if the
export-size limit is exceeded.

Initial beliefs, wealth, observations, and agent traits are identical across
network families. Within a network, payment rules share initial relationship
labels and recipient random draws. Their actual contacts and later relationships
can diverge when payment-dependent loyalty affects whom givers target. BA
matches `make_inputs` and `simulate` exactly at the requested horizon. The other
networks are new full-belief exploratory runs, extending the earlier cash-only
controls. Changing the horizon can change earlier contacts, so hold both seed and
horizon fixed for comparisons. A single realization is not an ensemble result.

### Belief Preservation Lab

Click **Belief Preservation Lab** in the observatory header, or open
`http://127.0.0.1:8765/strategy.html` while the same local server is running.
The original offline demonstration also links to this local page. This separate
experiment starts with a complete network and every pair mutually opposed.
Choose one payment rule per run; the existing observatory remains available for
multi-network and multi-rule comparisons.

The initial belief distribution is hierarchical. **Uniform** places the agents'
Gaussian centers evenly across `[0, 1]`; **Gaussian mixture** supports one to
three components with editable means, variances, and relative weights. Centers
are stratified across the bounded population distribution and assigned to agent
IDs by a seeded shuffle. A separate, shared **within-agent variance** controls
each agent's own Gaussian PMF on the theta grid. Gaussian input parameters are
not necessarily the achieved mean and variance after bounding and discretizing.
The mixture is conditioned on `[0, 1]`, so truncation can also change its effective
component proportions. Initial centers and exact PMFs are saved with the run.

Recipient types and payment strategies are separate controls. The three recipient
types are disjoint seeded draws: ordinary, **payment-dependent**, and **junzi**
(gentleman). The two configured fractions are sampling probabilities, so finite
populations need not contain exactly those proportions; the remainder is ordinary.

The model stores two relationship layers:

- **Persistent mutual signs:** a positive raw wealth-scaled tanh influence on an
  opponent pair creates a mutual alliance after simultaneous belief pooling.
  This test uses the opening base signs, before applying paid loyalty. Equality
  at the threshold is neutral. Once created, a base alliance persists until a
  manual sign assignment changes it (incitation remains inactive).
- **Effective directed responses:** ordinary recipients use the base sign.
  Payment-dependent recipients become allied toward each payer on any positive
  actual payment and reset that pair's unpaid clock. On the configured consecutive
  unpaid round, they become opposed before pooling, even if the base alliance
  remains. Rounds without contact also count as unpaid. Junzi ignore all incoming
  social influence and payment-driven sign changes, while public evidence and
  manual edits still apply. No payment can create a mutual base alliance on a pair
  involving a junzi. They may still give/receive cash and influence other people.

An entry `[j, i]` means **recipient i's response to payer j**. Effective signs
can therefore be asymmetric, while persistent base signs remain symmetric. The
network distinguishes mutual allies, mutual opponents, and mixed responses; the
relationship table shows both layers and both directions. At zero recipient cash,
tanh influence is zero, although a positive payment can still establish temporary
paid loyalty. Newly committed base signs cannot alter reverse same-round influence.

Choose among four recipient strategies, globally or through an individual behavior
intervention: **random one**, **all others**, **farthest**, or **farthest + random
other**. Farthest compares the payer's and recipient's current post-evidence PMFs
using Wasserstein-2; exact ties choose the lowest agent ID. The extra random
recipient is distinct from the payer and the farthest recipient. The amount rule
(fixed, fractional, hybrid, or zero) defines one **total budget per payer per round**,
which is split equally among selected recipients. Unaffordable total requests are
skipped. All strategies exclude self-payment and use simultaneous opening cash.
Saved uniforms drive random selections. Zero-payment contacts still occur and can
repel, so strategies differ in interaction count as well as money allocation.

The self-weight `a = (N - 1) * reference_payment * (1 - social_rate) / social_rate`
is inertia toward the agent's current post-evidence PMF, not actual spending or an
anchor to its original belief. The signed update before positive-part projection is
`z_i = q_i + sum_j w_ji * (q_j - q_i) / (a + sum_j abs(w_ji))`.
For entirely positive influences, the denominator normalizes a weighted average;
it sets self-influence relative to total incoming influence. With negative weights,
the same denominator limits the step scale, then negative PMF entries are set to
zero and the remaining masses normalized. The social-rate calibration gives a
self-share of `1 - social_rate` only if all `N - 1` neighbors contribute the
reference influence. It is not generally the realized mixing fraction.

The two proposed spending mechanisms are visible, editable, and saved, but
**inactive in this first version**. The proposed incitation rule would burn a
fixed cost paid by A to turn an allied B–C pair into opponents, without transferring
the money to an agent. The proposed self-reinforcement rule would burn `s`, set
`alpha = 1 - exp(-s / reinforcement_scale)`, and blend
`p_new = (1 - alpha) * p_current + alpha * p_initial`. Changing these cost controls
currently has no effect on trajectories or cash. There are no automatic actions
or paid-action intervention entries for these proposals yet.

Manual interventions remain active: change recipient type or payment strategy,
add or withdraw external cash, blend once toward a specified Gaussian PMF, or set
a mutual ally/opponent relationship. Entering payment dependence starts fresh
clocks from current responses; entering junzi freezes current responses; returning
to ordinary resumes the base signs. Reapplying an unchanged type does not reset
its clocks. A mutual sign edit changes both base and effective signs and resets
the pair's clocks. Connection removal is excluded so the graph stays complete.
Edits occur before their one-based update, in log order for ties. Payments or
loyalty expiry later in that round may change the edited responses again. Baseline
and edited runs share exact inputs; only the latter receives interventions.
Payments transfer money; total cash changes only through external adjustments.

**Save intervention log** saves only the ordered log as JSON. Loading it replaces
the current log and replays against this experiment's saved inputs. **Save experiment
JSON** also preserves initial PMFs, person types, evidence, and selection randomness.
**Save replay HTML** produces portable interactive playback; editing and loading
require the local server. CLI export accepts `"kind": "belief-preservation"` or its
saved experiment. Lab experiments and logs now use **format version 2**; version-1
files require the original implementation at commit `24633b9`, because their
relationship semantics differ. Source/dependency identity is checked separately
from the original observatory, whose model is unchanged.

The preservation metric is **Wasserstein-2 distance** from each agent's initial
PMF. Let `Q_p(u) = inf {theta: F_p(theta) >= u}` be the quantile function of a PMF
for `0 < u <= 1`. Then
`W_2(p, q) = sqrt(integral_0^1 [Q_p(u) - Q_q(u)]^2 du)`, and the preservation
score is `D_i(t) = W_2(p_i(t), p_i(0))`. For discrete PMFs, match probability mass
in increasing theta order, sum each matched mass times its squared displacement,
and take the square root. This evaluates the quantile integral exactly on the
specified grid.

Wasserstein-2 is the minimum root-mean-square displacement required to transport
one PMF into the other. It has theta units and lies in `[0, 1]` on this domain;
zero means identical PMFs. For example, changing a point mass at `0.2` to a
mixture with mass `0.75` at `0.2` and mass `0.25` at `0.6` gives
`sqrt(0.25 * (0.6 - 0.2)^2) = 0.2`. The page shows each agent's full PMF and
distance history, plus the population mean distance. Both preservation and
farthest-recipient selection use Wasserstein-2: target selection compares two
agents' current PMFs, while preservation compares an agent against its own
original PMF.

This measures preservation, not accuracy about the true bias. Public evidence
remains enabled by default; switch it off for a social-only experiment. These
strategies are candidate policies, not an optimal-strategy search. A useful next
experiment would minimize time-averaged Wasserstein-2 distance under a fixed total
budget, vary one focal agent's strategy against random background agents, and
compare paired seeds before evaluating on fresh seeds.

### Batch experiments

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m bayes_cash --output outputs/demo
python -m pytest -q
```

The command runs **both uniform and loguniform initial wealth**, each with fixed,
fractional, hybrid, and no-payment scenarios. With the default traits and all
allies, the no-payment control learns only from public evidence. With opponents
it also experiences undamped repulsive contacts. A nonempty output directory is
refused to protect previous runs.

For another paired realization:

```bash
python -m bayes_cash --seed 43 --rounds 1000 --output outputs/seed43
```

To give each undirected edge a 20% probability of being an opponent relationship:

```bash
python -m bayes_cash --opponent-fraction 0.2 --output outputs/opponents_demo
```

To sample 40% ally-preferring givers, 40% opponent-preferring givers, and 20%
uniform givers, with an independent 30% probability of requiring regular payment:

```bash
python -m bayes_cash --opponent-fraction 0.2 \
  --ally-giver-fraction 0.4 --opponent-giver-fraction 0.4 \
  --preference-probability 0.8 --conditional-ally-fraction 0.3 \
  --loyalty-rounds 5 --output outputs/giver-loyalty-demo
```

The fractions are sampling probabilities, so realized population shares vary.
Set `--preference-probability 1` for strict preferred-group targeting. Both giver
fractions and the conditional-ally fraction default to zero, retaining the
original uniform-contact, static-relationship model exactly.

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
social rate mu = 0.25. All agents are persuadable; there are no partisans. The
default opponent fraction is zero. For a positive fraction, each existing edge
is independently designated ally or opponent by a separate random stream. These
initial labels are symmetric. They remain fixed unless payment-dependent
loyalty is enabled, which can make later relationships directional. The optional
tanh rule separately lets effective contact weights change sign. Adding traits
or relationship labels changes none of the random graph, prior, wealth, or coin
draws; realized contacts can depend on relationships when giver preferences are
enabled.
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

By default, each payer j chooses one neighbour r_j uniformly, independently of
opinions, cash, and affordability. The original uniform recipient schedule is
generated for **all rounds** before any scenario runs. Optional giver traits
instead use shared random draws to choose from the current relationship groups:

- `ally_giver_fraction`: probability an agent prefers paying allies.
- `opponent_giver_fraction`: probability an agent prefers paying opponents.
  The two fractions sum to at most one; the remainder choose uniformly over all
  neighbours.
- `preference_probability` (default 0.8): relative group weight p for the
  preferred group and 1-p for the other group, then uniform selection within the
  chosen group. With both groups present, the preferred group is chosen with
  probability p regardless of its size; p=0.5 gives equal group probabilities,
  not uniform selection over all neighbours when the groups differ in size.
  Weights are normalized over nonempty groups with positive weight. At p=1,
  only the preferred group is eligible; if
  it is empty the giver skips contact and payment. At p=0, the same rule applies
  to the other group. Uniform givers are unaffected by p.

The directed sign `relationships[j, i]` means **recipient i's receptivity to
payer j**: +1 for ally and -1 for opponent. A giver selects recipients receptive
or opposed to that giver; the reverse direction need not have the same sign.
Selection uses relationships at the start of the round, before current
payments or loyalty expiration. It never chooses by the payment's eventual
effective influence under tanh. Shared random draws preserve pairing, but
payment rules can produce different recipient choices after their relationships
diverge.

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

An agent with no eligible recipient pays zero. Every payment is debited from its
payer and credited to its recipient, so total cash is conserved. Simultaneous
settlement prevents agent iteration order from changing who can afford to pay.

### Payment-dependent loyalty

`conditional_ally_fraction` (default 0) independently samples which recipients
require regular payments. This trait can coexist with either giver preference
or uniform giving. `loyalty_rounds` is a positive integer, default 5.

For a conditional recipient i, any strictly positive **actual** payment from j
immediately sets `relationships[j, i] = +1` and resets that pair's unpaid-round
counter to zero. The reverse relationship is unaffected. Counters advance every
simulation round without a positive j-to-i payment, including rounds when j
contacts somebody else, skips contact, cannot afford the payment, or requests
zero. On the fifth consecutive unpaid round (or the configured threshold), i
becomes j's opponent. Each payer has a separate counter, so payment from one
payer cannot sustain loyalty to another.

All pairs begin with the sampled initial signs and zero unpaid rounds. Thus
initial conditional allies also defect after five unpaid rounds. Ordinary
recipients retain their initial signs. Loyalty updates occur after payment
amounts are determined and before social influence: a payment that converts an
opponent uses the allied influence law in the same round; an unpaid contact on
the expiration round uses the opponent law. Cash and belief updates remain
synchronous. The tanh law uses this updated relationship sign and the
recipient's opening wealth.

### 4. Payment-weighted social pooling: allies-only limit

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

### 5. Opponents: payment weakens repulsion

For each scheduled contact j to i, let s_ji=+1 denote an ally and s_ji=-1 an
opponent. Apply the following cash-valued influence weight to its actual payment:

```math
w_{ji}(m_j)=\begin{cases}
m_j,&s_{ji}=+1,\\
-r(m_j),&s_{ji}=-1,
\end{cases}
\qquad
r(m)=\frac{m_{\rm ref}}{1+m/m_{\rm ref}}.
```

The repulsive weight is largest at zero payment, is halved at m=m_ref, and tends
to zero as payment grows. It never becomes attractive. The same reference amount
calibrates allied influence and the attenuation of opponent influence; this
avoids adding an independently uncalibrated scale in the first signed experiment.

Using the same just-Bayesian-updated distributions q and the same K_i, define

```math
u_i(k)=q_i(k)+
\frac{\sum_{j:r_j=i}w_{ji}(m_j)[q_j(k)-q_i(k)]}
{K_i+\sum_{j:r_j=i}|w_{ji}(m_j)|},
\qquad
p_i'(k)=\frac{\max[0,u_i(k)]}{\sum_\ell\max[0,u_i(\ell)]}.
```

The raw distribution u_i sums to one because each difference q_j-q_i sums to
zero. Signed differences can nevertheless make individual masses negative. The
positive-part normalization is an explicit assumption inherited from the source
model's signed social step, not a numerical repair. It can eliminate support for
a hypothesis; later Bayesian evidence alone cannot revive that support, although
allied contacts can. No probability floor is introduced, and zero predictive
probability still raises an error.

For implementation, let A_i be total allied cash received, B_i the sum of the
positive repulsive weights, and D_i=K_i+A_i+B_i. Expanding the numerator gives

```math
u_i=\frac{(K_i+2B_i)q_i+\sum_{j:r_j=i}w_{ji}q_j}{D_i}.
```

For B_i=0 this is exactly the existing allied pooling expression; the code keeps
its numerical ordering and does not apply an unnecessary extra normalization.
The allies-only baseline is therefore retained bit for bit. With a single
opponent of weight r, u_i=q_i+[r/(K_i+r)](q_i-q_j), which moves away from the
opponent's distribution. Increasing payment reduces that repulsive coefficient.
With several contacts, final belief displacement need not be monotonic because
other contributions and positive-part normalization also change.

**Unpaid contacts:** each scheduled contact is observed even if its payment is
zero or unaffordable. An unpaid ally has zero influence; an unpaid opponent has
maximum repulsion. This is the new information assumption needed for monotonically
decreasing repulsion from zero payment. Consequently, the mixed-network
no-payment control combines public evidence with undamped antagonism. It is not
the allies-only evidence control.

Cash is transferred identically along allied and opposing edges, and the sign
does not change affordability. With uniform givers, wealth and payment histories
remain identical between all-allied and mixed networks. Preference-based
targeting can change contacts and therefore later cash balances. The spending
amount is still imposed by the selected payment rule rather than derived from
utility maximization.

### 6. Optional wealth-scaled tanh transition

```bash
python -m bayes_cash --influence-rule tanh --opponent-threshold 0.05 \
  --opponent-fraction 0.2 --output outputs/tanh_demo
```

The confirmed alternative assigns a directed contact from payer j to recipient i
the weight

```math
w_{ji}(t)=W_i(t)\tanh\!\left[\frac{m_j(t)}{W_i(t)}
                  -\kappa\,\mathbf{1}_{s_{ji}<0}\right],
\qquad \kappa=0.05.
```

Both the leading amplitude and the money scale inside tanh are the recipient's
**opening wealth**, before current income or transfers. For a current opponent,
payments below 5% of that wealth repel, equality is neutral, and larger payments
attract. The sign is recalculated each contact and can switch back later. This
does not itself store a cumulative relationship or change the relationship
label; payment-dependent loyalty separately changes that label. Current allies
remain nonnegative; an unpaid allied contact has zero weight.
When recipient wealth is exactly zero, the prefactor limit sets the weight to
zero without an epsilon. Receiving cash can then affect subsequent rounds.

With f=kappa=0.05, a fractional payment makes a current opponent attractive
exactly when the payer is wealthier than the recipient: its tanh argument is
`0.05*(W_j/W_i-1)`. For a fixed payment c=1, attraction instead requires
recipient wealth below `c/kappa=20`, provided the payer makes the payment.
The implementation evaluates the equivalent argument `(m-kappa*W_i)/W_i`
for opponents, preserving exact neutrality when payment equals the threshold
even at very large wealth. No numerical tolerance changes the sign boundary.

The existing signed social equation is used with these effective weights.
Repulsion and positive-part projection are determined by the **effective
negative weights**; attractive weights enter the positive influence term even
for a contact whose relationship label is negative. The old constant self-weight remains
`degree * reference_payment * (1-social_rate)/social_rate`; `reference_payment`
continues to set this fixed money scale but does not set the tanh denominator.
For small relative allied payments, `W_i*tanh(m/W_i)` approaches m, recovering
the old allied contact weight to first order.

Keeping self-weight fixed confines the intervention to the contact law, but has an important
consequence when income is enabled: proportionally growing wealth and payments
increase tanh weights relative to self-belief. The new model is not invariant
under rescaling all nominal wealth while holding self-weight fixed. A model
with wealth-scaled self-weight would be a different convention and is not used
in this comparison. The old no-allied-neighbour singleton certificate must not
be applied to tanh or payment-dependent-loyalty runs: initial opponents can
supply attractive input later.

### 7. Optional proportional income and an income budget

```bash
python -m bayes_cash --income-rate 0.1 --opponent-fraction 0.2 \
  --output outputs/income_demo
python -m bayes_cash --income-rate 0.1 --influence-rule tanh \
  --opponent-fraction 0.2 --output outputs/tanh_income_demo
```

With positive income rate r, agents earn `I_i(t)=r*W_i(t)` from opening wealth.
The existing fixed, fractional, or hybrid rule first makes a payment request
g_i. It is paid only if affordable from opening cash **and** no larger than the
current income budget; otherwise the whole request is skipped. Requests are
never reduced to fit the budget, preserving the fixed/hybrid minimum.

```math
m_i=g_i\,\mathbf{1}_{g_i\leq W_i}\,\mathbf{1}_{g_i\leq rW_i},
\qquad
W_i(t+1)=(1+r)W_i(t)-m_i(t)+\sum_{j:r_j=i}m_j(t).
```

Income and transfers settle simultaneously. Receipts from other agents cannot
finance the current request. At r=0.1, f=0.05, c=1, fractional payments always
fit the income budget; fixed and hybrid payments require opening wealth at
least 10. Every agent retains at least its opening wealth after income and its
own spending, even before incoming transfers. This is an imposed budget that
protects principal, not a derivation of rational or utility-maximizing behaviour.
Setting `income_rate=0` disables this additional constraint and reproduces the
original no-income model, which permits spending existing wealth.

Transfers conserve money, while income is an external source:

```math
S(t+1)=(1+r)S(t),\qquad S(t)=S(0)(1+r)^t.
```

Exports retain **nominal** wealth, payments, and income. Cash validation compares
`S(t)/(1+r)^t` with the initial total rather than incorrectly demanding constant
nominal cash. The 10% is per simulation round, not an empirically calibrated
annual income. It produces enormous nominal growth over thousands of rounds.
The current simulator explicitly rejects horizons that exceed finite nominal
floating-point money; 20,000 rounds at r=0.1 are unsupported. A longer income
study needs a wealth-share/log-total representation that also retains the
fixed payment as a fixed nominal amount.

The tanh law also changes repulsion strength even without income: at zero
payment the old weight is -1, while the new weight is
`-W_i*tanh(0.05)`, approximately `-0.05*W_i`. Thus the comparison below tests the
complete proposed law, not the isolated causal effect of permitting sign flips.

## Saved data and diagnostics

Each run saves:

- `summary.json`: parameters, installed package versions, upstream revision,
  conventions, final diagnostics, and numerical conservation errors.
- `{loguniform,uniform}_trajectories.npz`: grid, network, initial relationship signs,
  sampled agent traits, initial distributions, initial wealth, shared evidence,
  recipient random inputs, and full belief/wealth/payment/actual-recipient/
  directed-relationship histories for all four scenarios.
- `{loguniform,uniform}_metrics.csv`: opinion mean, disagreement, uncertainty,
  mean squared belief error, wealth summaries, Gini, total payments, and the
  fraction of agents who paid in each round.
- `{loguniform,uniform}_comparison.png` and `.pdf`: six time-series diagnostics.
- `wealth_redistribution.png` and `.pdf`: initial versus final wealth, with
  network degree indicated by colour.

For a scenario, beliefs have shape (T+1,N,K) and wealth (T+1,N), including the
initial state at index 0. Payments and actual recipients have shape (T,N), and
directed relationships have shape (T+1,N,N). Transaction `payments[t,j]` moves
from j to that scenario's `recipients[t,j]` and produces state t+1. Recipient -1
means no eligible contact and always has zero payment. Relationship state t+1
includes that round's conversions and expirations and is used for its social
update. This is the complete transaction ledger, including attempted contacts
with zero payment. The unprefixed `recipients` field retains the original
uniform schedule; use `{mode}_recipients` to replay an actual scenario.
`giving_preferences` stores 1 (allies), -1 (opponents), or 0 (uniform);
`conditional_allies` stores the independent recipient trait. `contact_uniforms`
stores preference-selection random inputs when those are present.
No pickle is needed to read the archives:

```python
import numpy as np

with np.load("outputs/giver-loyalty-demo/loguniform_trajectories.npz") as data:
    wealth = data["fractional_wealth"]
    payments = data["fractional_payments"]
    recipients = data["fractional_recipients"]
    relationships = data["fractional_relationships"]
    mean_opinions = data["fractional_beliefs"] @ data["theta"]
```

Diagnostic belief MSE is the population average of
sum_k p_i(k)(theta_k-theta_true)^2; it includes both squared mean error and
within-agent uncertainty. Disagreement is the standard deviation of agent means,
which is a different quantity. All figures are exploratory diagnostics.

### Included demonstration

`outputs/demo` contains the original all-allied 100-agent, 500-round run with seed 42. Its
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

The analysis and included study results below use uniform givers and fixed
relationship labels. The four historical study scripts (`check_network_effects`,
`compare_opponents`, `sweep_opponents`, and `compare_extensions`) explicitly
reject nonzero giver-preference or conditional-ally fractions: their cash
pairing, static-sign diagnostics, or chunked continuation assume that scope.
Use the canonical batch command or explorer for the new traits. Existing saved
results remain unchanged.

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

Run the cash subsystem for the original uniform-contact, no-income model:

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

## Paired opponent comparison

```bash
python scripts/compare_opponents.py --seeds 20 --rounds 500 \
  --opponent-fraction 0.2 --output outputs/opponent_comparison
```

The included comparison uses 100 agents, seeds 42--61, c=1, f=0.05, and a 20%
opponent-edge probability. The realized fraction is 0.1967 +/- 0.0258 across
seeds. Each mixed-network run is paired with an all-allied run having exactly
the same graph, initial beliefs, wealth, evidence, and contacts. All three
payment rules and both no-payment controls are included. The independently
audited signed update is documented above; no run failed or was excluded.

Final disagreement means the standard deviation across agents' posterior means.
The values below are the average across seeds, with sample SD for the mixed arm:

| Wealth | Payment | Allies-only disagreement | With opponents, mean +/- SD |
| --- | --- | ---: | ---: |
| Loguniform | Fixed | 6.01e-6 | 0.01865 +/- 0.00226 |
| Loguniform | Fractional | 6.05e-8 | 0.00762 +/- 0.00291 |
| Loguniform | Hybrid | 7.45e-9 | 0.00540 +/- 0.00240 |
| Uniform | Fixed | 2.99e-6 | 0.01668 +/- 0.00205 |
| Uniform | Fractional | 1.74e-11 | 0.000952 +/- 0.001702 |
| Uniform | Hybrid | 1.55e-11 | 0.000944 +/- 0.001692 |

The no-payment control has disagreement 0.001123 +/- 0.000372 with all allies
and 0.02726 +/- 0.00396 with opponents; it is identical across initial wealth
distributions because neither control spends money. Remember that the mixed
no-payment control includes unpaid repulsion.

All-allied paid belief MSEs are about 0.001079. In the loguniform mixed runs,
fixed/fractional/hybrid MSEs are 0.001532/0.001149/0.001101. In the uniform mixed
runs they are 0.001403/0.001082/0.001082. These measure error as well as uncertainty,
not just agreement. Full per-seed values and paired differences are saved, so
overlapping marginal error bars need not be used to assess a paired contrast.

In this experiment, fractional and hybrid payments suppress opponent-driven
disagreement more effectively than fixed payments. Under the fixed rule,
increasing cash concentration leaves many contacts unpaid, at maximum repulsion,
and even a rich opponent never pays more than c. Fractional spending limits
that accumulation and allows larger payments to weaken repulsion further. The
hybrid's floor can strengthen allied attraction and opponent attenuation among
solvent agents whose fractional payments would be small. The net result remains
network- and parameter-dependent: these runs do not isolate attenuation from
attraction, nor equate realized spending across rules.

For loguniform wealth the hybrid has lower average disagreement than fractional
payments despite higher wealth Gini (0.580 versus 0.369). For uniform wealth their
belief results are almost indistinguishable. The greater typical wealth in the
uniform draws means the minimum-payment floor is active less often.

Every wealth and payment history was verified **bitwise identical** between the
two relationship arms. Cash conservation error is below 7.3e-12 cash units.
The complete original demo was also reproduced bit for bit with the new core,
and its committed files remain unchanged. `outputs/opponents_demo` contains the
seed-42 signed run and full trajectories. `outputs/opponent_comparison` contains
the paired ensemble's figures, per-seed CSV, summary JSON, and replayable inputs
plus metric histories in NPZ. These are 500-round exploratory results at one
opponent probability; no stationary or empirically calibrated claim is made.

## Opponent-fraction sweep and long-run check

```bash
python scripts/sweep_opponents.py --seeds 20 --rounds 20000 --workers 6 \
  --output outputs/opponent_sweep
python scripts/plot_opponent_sweep.py outputs/opponent_sweep
```

The sweep uses opponent-edge probabilities 0, 0.1, ..., 1, both wealth
distributions, and all three payment rules. Each seed retains the same topology,
priors, wealth quantiles, observations, and contact schedule across fractions and
payment rules. Thresholding the same edge uniforms makes the opponent sets
nested as the probability increases. The plotted horizontal coordinate is the
assignment probability; realized edge fractions are saved separately.

The first 500 rounds reproduce the earlier comparison exactly. Independent
seeded continuation streams extend these inputs to round 20,000; time-major
contact draws also preserve the continuation when a still longer horizon is
requested. Checkpoints at 500, 5,000, and 20,000 belong to the same trajectories.
The runner calls the existing simulator in bounded chunks, without changing
the equations, and checks cash-path hashes across every opponent fraction.

Disagreement is the population standard deviation D of agents' posterior means
mu_i. Accuracy is measured by the existing posterior belief squared error:

```text
belief MSE = mean_i Var_i(theta) + mean_i (mu_i - theta_true)^2
           = mean_i Var_i(theta) + D^2 + (mean_i mu_i - theta_true)^2.
```

The decomposition distinguishes uncertainty, disagreement, and a shared error.
The sweep also records exact zero mass at the true grid point, mean mass there,
and singleton beliefs. No probability floor is introduced. The true bias must
lie on the grid for the support diagnostics to be defined.

Late-window comparisons average every round from 10,001--15,000 and from
15,001--20,000 within each seed, then compute the seed mean and sample SD. These
are exact window averages, independent of the stride used to save time series.
Across-seed bands show sample SD, not confidence intervals. Agreement between
late windows is evidence of persistence over this run, not by itself a proof
of a stationary state.

There is also an exact obstruction to future consensus. An agent with no allied
neighbours and a singleton belief at an interior grid point keeps that belief
under both Bayesian learning and repulsive social updates. For a set F of such
agents, all future disagreement has the lower bound

```text
D_future >= sqrt(sum_{i in F} (theta_i - mean_F(theta))^2 / N).
```

The bound follows by allowing every remaining agent to move to the mean of the
fixed agents, which minimizes population variance. Distinct fixed beliefs make
the bound positive. It is conditional on retaining the current finite-grid
model, projection rule, and relationship network; adding a belief floor or
changing the network would change this conclusion.

### Included sweep results

All 1,320 trajectories (20 seeds x 11 probabilities x two distributions x three
rules) completed without failed or excluded runs. The 240 old checkpoints at
p=0 and p=0.2 reproduce the previous 500-round metrics exactly. Complete wealth
and payment paths match across opponent fractions; maximum cash-conservation
error is 1.6e-10 cash units. The implementation suite has 88 passing tests.

Selected ensemble-mean disagreement values at round 20,000:

| Initial wealth | Opponent probability | Fixed | Fractional | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Loguniform | 0 | approximately 0 | approximately 0 | approximately 0 |
| Loguniform | 0.2 | 0.02211 | 0.007219 | 0.004294 |
| Loguniform | 0.5 | 0.03579 | 0.02967 | 0.02651 |
| Loguniform | 1 | 0.04756 | 0.04497 | 0.04325 |
| Uniform | 0 | approximately 0 | approximately 0 | approximately 0 |
| Uniform | 0.2 | 0.02142 | 0.001492 | 0.001492 |
| Uniform | 0.5 | 0.03523 | 0.01664 | 0.01661 |
| Uniform | 1 | 0.04625 | 0.03890 | 0.03887 |

Corresponding belief MSEs (smaller is better):

| Initial wealth | Opponent probability | Fixed | Fractional | Hybrid |
| --- | ---: | ---: | ---: | ---: |
| Loguniform | 0 | approximately 0 | approximately 0 | approximately 0 |
| Loguniform | 0.2 | 1.261e-3 | 4.315e-4 | 2.792e-4 |
| Loguniform | 0.5 | 1.709e-3 | 1.482e-3 | 1.371e-3 |
| Loguniform | 1 | 3.193e-3 | 2.908e-3 | 2.716e-3 |
| Uniform | 0 | approximately 0 | approximately 0 | approximately 0 |
| Uniform | 0.2 | 1.247e-3 | 7.500e-6 | 7.500e-6 |
| Uniform | 0.5 | 1.713e-3 | 5.299e-4 | 5.215e-4 |
| Uniform | 1 | 3.019e-3 | 2.228e-3 | 2.221e-3 |

Here the all-allied disagreement is about 1e-16 and belief MSE about 2e-30:
effectively agreement on the true point of this discrete grid. Fixed payments
have the largest ensemble-mean final disagreement and MSE at every positive
tested probability, for both distributions. Hybrid usually improves upon
fractional payments for loguniform wealth; their uniform-wealth results are
close. This is not a universal ordering: at loguniform p=0.1 the fractional
endpoint disagreement is slightly lower, whereas the hybrid's final-window
average is lower. At uniform p=0.6 the hybrid's endpoint MSE is slightly higher.
The wealth distributions also differ in typical and total wealth, hence in
payment burdens; their contrast does not isolate distribution shape.

**Longer learning does not generally remove disagreement.** At p=1, the changes
in ensemble-mean disagreement between the two late windows are below 0.2%, and
belief MSE is almost flat. More strongly, every one of these 120 all-opponent
runs contains permanently fixed agents at distinct interior grid points. Their
mean lower bounds on future disagreement range from 0.0176 to 0.0202 across the
six distribution/payment combinations. These are guaranteed lower bounds, not
predictions of the eventual disagreement.

Across the whole sweep, 715/1,320 runs have a resolved positive certificate at
round 20,000 (387 loguniform, 328 uniform). A 1e-12 guard excludes floating-point
roundoff from this count; the smallest resolved positive bound is 0.0035355.
Every seed and payment rule is certified at the tested probabilities p>=0.6 for
loguniform wealth and p>=0.8 for uniform wealth. These observations do not
locate a phase boundary. At p=1, 56.4--64.1% of agents assign exactly zero mass
to the truth, depending on the distribution and payment rule. Repulsion and its
positive-part projection can eliminate hypotheses that subsequent Bayesian
conditioning cannot recover without allied input.

**Some small residuals remain transient.** At loguniform p=0.1, fractional
late-window disagreement decreases from 0.000728 to 0.000469, and hybrid from
0.000461 to 0.000407; their MSEs also decrease. These runs have not established
a nonzero asymptotic limit. Even where disagreement looks steady, individual
runs and accuracy can fluctuate: loguniform p=0.2 hybrid mean MSE rises from
8.20e-5 to 1.09e-4 between late windows. More observation is therefore not a
guarantee of monotonically improving accuracy under signed social influence.

Seed heterogeneity matters. At uniform p=0.2, both fractional and hybrid have
14/20 seeds at numerical agreement on truth. The other six retain one agent
concentrated on 0.55 or 0.65, giving disagreement 0.004975 and MSE 2.5e-5 within
each such run. Their ensemble disagreement is 0.001492. This residual is stable
over both late windows, but their tiny surviving belief tails mean they do not
meet the exact-singleton certificate. Neither every seed nor every nonzero
endpoint should be described as permanently polarized.

Disagreement need not increase monotonically with opponent probability, and
less disagreement need not mean greater accuracy. For example, uniform
fractional payments give disagreement 0.03952 at p=0.9 and 0.03890 at p=1, while
MSE increases from 0.002128 to 0.002228.

The saved figures answer complementary questions:

- [Disagreement sweep](outputs/opponent_sweep/disagreement_sweep.png): how
  population spread changes across probability and the three horizons.
- [Belief accuracy sweep](outputs/opponent_sweep/belief_accuracy_sweep.png): the
  same comparison including uncertainty and error about the true bias.
- [Late persistence](outputs/opponent_sweep/late_persistence.png): exact averages
  from the last two 5,000-round windows, with sample SD across seeds.
- [Truth support and absorbing bounds](outputs/opponent_sweep/absorbing_support.png):
  current exclusion of truth versus a certified obstruction to future consensus.

Each figure also has a PDF. `per_seed.csv` and `late_windows.csv` contain the
individual results; `summary.json` contains group statistics, conventions, and
cash hashes. `trajectories.npz` stores diagnostics every 50 rounds and all
checkpoints, while `inputs.npz` stores the full replayable random inputs.
The 21-point grid and support-erasing projection remain scientific assumptions;
no grid-refinement or alternative-projection study is included here.

## Separate tanh and income comparison

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  python scripts/compare_extensions.py --seeds 20 --rounds 5000 \
  --horizons 500 5000 --fractions 0.2 1 --workers 6 \
  --output outputs/extensions
python scripts/plot_extensions.py outputs/extensions
```

Four paired arms separate the additions: baseline attenuation without income,
tanh only, income with the original attenuation law, and both additions.
The 960 trajectories use 20 seeds, both initial wealth distributions, all three
payment rules, and opponent probabilities 0.2 and 1. These two probabilities
probe mixed and entirely opposing networks; this extension is not a repeat of
the full 0--1 sweep. Parameters remain N=100, c=1, f=0.05, kappa=0.05, and
income rate 0.1 when enabled. Each arm shares the same graph, beliefs, wealth
quantiles, evidence, and contacts. Checkpoints at 500 and 5,000 are from the
same trajectory; all 480 baseline checkpoint rows exactly reproduce the prior
sweep for the five shared belief/wealth metrics.

Mean posterior belief MSE at round 5,000 (smaller is more accurate):

| Wealth | Opponents | Payment | Baseline | Tanh only | Income only | Both |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Loguniform | 20% | Fixed | .001241 | .001379 | .000479 | .002742 |
| Loguniform | 20% | Fractional | .000260 | .001377 | 4.44e-10 | .001335 |
| Loguniform | 20% | Hybrid | .0000716 | .001593 | 4.44e-10 | .001345 |
| Uniform | 20% | Fixed | .001248 | .001860 | .000470 | .002626 |
| Uniform | 20% | Fractional | .00000750 | .002180 | 4.46e-10 | .001359 |
| Uniform | 20% | Hybrid | .00000750 | .001843 | 4.46e-10 | .001362 |
| Loguniform | 100% | Fixed | .003189 | .001527 | .002921 | .006849 |
| Loguniform | 100% | Fractional | .002906 | .009456 | .0000838 | .025449 |
| Loguniform | 100% | Hybrid | .002715 | .010115 | .0000913 | .025274 |
| Uniform | 100% | Fixed | .003011 | .002216 | .002934 | .008283 |
| Uniform | 100% | Fractional | .002226 | .015536 | .0000150 | .026494 |
| Uniform | 100% | Hybrid | .002218 | .015354 | .0000150 | .025733 |

Income alone greatly improves fractional/hybrid accuracy under the original
attenuation law. At 20% opponents, disagreement falls to 2.28--2.91e-11 across
these four wealth/payment combinations. Growing payments suppress the original
opponent weight toward zero, while increasing allied pooling. At 100% opponents,
income-only fractional/hybrid errors are also much smaller, but previous support
loss can still prevent full recovery by evidence alone.

The tanh law does not share that implication: a payment below the recipient's
relative threshold remains repulsive, and its magnitude grows with recipient
wealth. Tanh alone improves fixed payments in entirely opposing networks but
worsens fractional/hybrid accuracy. With both additions, the latter MSE reaches
0.0253--0.0265 at 100% opponents despite disagreement of only 0.0315--0.0475.
The per-seed MSE decomposition attributes 75--86% of that error to squared
population-mean bias. Oppositely biased seeds can cancel in an ensemble mean,
so a mean opinion near the truth would not establish accurate agreement.

For both additions, fractional/hybrid disagreement still decreases by about
10--13% between the exact late windows 2,501--3,750 and 3,751--5,000, while MSE
remains about 0.025--0.028 in entirely opposing networks. These are finite-time
results, not stationary or certified frozen states. Constant self-weight makes
income amplify tanh social strength relative to self-belief; changing that
convention requires a separate experiment.

Cash histories are identical across influence laws and opponent probabilities
at each income setting. In wealth shares x_i=W_i/S, unrestricted fractional
payments with income obey

```math
x_i'=x_i+\frac{f}{1+r}\left(\sum_{j:r_j=i}x_j-x_i\right).
```

Thus proportional income alone preserves relative inequality, while these
transfers act like the original share dynamics at fraction f/(1+r). Fixed
payments become negligible relative to the exponentially growing total S,
leaving Gini near its initial value instead of concentrating wealth to the
no-income value near 0.96. With income, the final fixed Gini is 0.568 for
loguniform and 0.328 for uniform wealth; fractional and hybrid are both 0.369.
The hybrid becomes fractional once every opening balance exceeds c/f=20.

The saved figures answer the comparison questions directly:

- [Beliefs at round 500](outputs/extensions/belief_comparison_500.png) and
  [round 5,000](outputs/extensions/belief_comparison_5000.png): disagreement and
  posterior MSE for all four arms, using common scales and seed sample SDs.
- [Income and cash](outputs/extensions/income_cash.png): inequality and payment
  participation over time, with and without income.
- [Effective contact signs](outputs/extensions/contact_signs.png): the fraction
  of scheduled baseline-opponent contacts that actually attract. Fractional
  payments flip about 30--32% at round 5,000; fixed payments with income flip
  none because recipients have grown beyond the c/kappa=20 threshold.

Each figure also has a PDF. `per_seed.csv` and `late_windows.csv` preserve
individual outcomes; `summary.json` records parameters, replay conventions,
cash hashes, means, and sample SDs. The compact `trajectories.npz` saves
diagnostics every 50 rounds. All 960 runs succeeded; maximum relative error
in the income-adjusted cash identity is 4.14e-13. At the time of this study, the
implementation suite passed 129 tests, including threshold signs, income budgets,
simultaneous accounting, unchanged baseline trajectories, and large-value
summary statistics. These checks establish numerical contracts, not model
validity or convergence.

## Saved artifacts and version control

Git retains the code, tests, small figures and CSV/JSON results, and
`outputs/opponent_sweep/trajectories.npz`. This archive contains diagnostics
from 1,320 trajectories of 20,000 rounds and is costly to reproduce. The same
retention policy includes `outputs/extensions/trajectories.npz` (about 3 MB),
whose diagnostics require 960 trajectories of 5,000 rounds to regenerate.

Other NPZ archives are ignored: the single-seed demos, the shorter paired
comparison, the cash-only network controls, and the sweep's raw `inputs.npz`.
They remain available locally when generated. The commands in the corresponding
sections regenerate the simulation outputs. Sweep inputs alone can be rebuilt
with `make_sweep_inputs` in `scripts/sweep_opponents.py`, using the configuration,
seeds, and opponent fractions recorded in `summary.json`; generating them does
not require running the belief dynamics. Cheap archives committed previously
remain in historical commits, but are no longer tracked in the current tree.

## Interpretation limits and next decisions

- The model has **wealth-to-opinion coupling**. Spending and recipient
  choice do not depend on beliefs, so changing beliefs cannot change wealth.
  Cash and payment-dependent loyalty jointly evolve independently of beliefs;
  relationship-based targeting can feed past payments back into later contacts.
  Optional income is exogenous and proportional to wealth;
  there are no wages from labour, production, goods, or utility optimisation.
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
- Giver preferences, loyalty duration, opponent attenuation, the opponent
  fraction, and the positive-part response require scientific sensitivity
  studies. The first signed comparison uses a
  disclosed 20% opponent-edge probability and the shared reference payment scale;
  these choices have not been empirically calibrated.

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
