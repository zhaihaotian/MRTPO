# Advantage estimators

All three are in `vendor/verl/verl/trainer/ppo/ray_trainer.py::compute_advantage`; MRTPO's replay, weights and bound are
in `vendor/verl/verl/trainer/ppo/mrtpo.py`.

Notation: response j of prompt i, G = 4 responses per prompt, rewards k ∈ {format, correctness}.
$Z_{ijk} = (r_{ijk} - \mathrm{mean}_j r_{ijk}) / (\mathrm{std}_j r_{ijk} + 10^{-6})$ is the group z-score with the
sample standard deviation. "Whitening" is the final normalization over all valid response tokens of the batch: subtract
the mean, divide by $\sqrt{\mathrm{var} + 10^{-8}}$.

| Method | `--method` | Token advantage |
|---|---|---|
| GRPO | `grpo` | z-score of the summed reward after the reward-KL penalty, $r_{ij} = r_{ij,\mathrm{format}} + r_{ij,\mathrm{correctness}} - 0.001 \sum_t \mathrm{KL}_t$, same value on every token; no whitening |
| GDPO | `gdpo` | $Z_{ij,\mathrm{format}} + Z_{ij,\mathrm{correctness}}$ on every token, then whitening |
| MRTPO | `mrtpo` | $\sum_k Z_{ijk}\, w_{ijk,t}$, optionally bounded (below), then whitening |

## MRTPO

**Replay.** For response $y$ of a prompt, four contexts are built from the rollout prompt $P_0$ by appending to the user
message the information a reward depends on (`vendor/verl/verl/utils/rlla_replay.py`):

| Context | Added to the user message |
|---|---|
| $P_F$ | the response form the format reward checks (tool-call block, or direct answer) |
| $P_C$ | the reference calls: operation order, names, parameter names and values (from `extra_info.teacher_output`) |
| $P_{FC}$ | both blocks |

The policy scores the same sampled tokens under two contexts and, at every response position $t$, computes the
Jensen–Shannon divergence between the two next-token distributions over the full vocabulary:

- correctness: $\mathrm{JS}_t(P_{FC}, P_F)$, the effect of the reference calls (tool-call rows; rows whose label is a
  direct answer have no correctness reward);
- format: $\mathrm{JS}_t(P_{FC}, P_C)$ on tool-call rows, $\mathrm{JS}_t(P_F, P_0)$ on direct-answer rows.

The replay runs only for (response, reward) pairs whose prompt group varies on that reward: when all G responses score
the same, $Z_{ijk} = 0$ exactly, so the weights could not change anything and those forward passes are skipped.

**Weights.** $q_t = \sqrt{\mathrm{JS}_t / \ln 2} \in [0, 1]$ and

$$
w_t = (1 - b) + b\,\frac{q_t}{\bar q},\qquad b = \beta \cdot s(\mathrm{step}),\qquad s = 3p^2 - 2p^3,\; p = \min(\mathrm{step}/K, 1),
$$

with $\bar q$ the mean of $q$ over the response's tokens; the weights are divided by their response mean, so every
response keeps its GDPO advantage on average. A response whose $q$ is zero everywhere keeps $w_t = 1$. β = 0.5 and
K = 10 for both rewards; there is no weight cap. With β = 0 MRTPO is exactly GDPO (the CPU tests check this).

**Reallocation bound** (on by default, ρ = 1). Let $B_{ij,t} = \sum_k Z_{ijk}$ (the GDPO token advantage) and
$R = \sum_k Z_{ijk} w_{ijk,t} - B$ the reallocation. If, over all valid tokens of the batch,
$\overline{R^2} > \rho\, \overline{B^2}$, then $R$ is multiplied by $\sqrt{\rho\, \overline{B^2} / \overline{R^2}}$. One
common factor keeps every response's mean advantage and its token ordering. The factor is logged as
`mrtpo/reallocation_scale`.

## Logged quantities

`mrtpo/b` (current mixture strength), `mrtpo/reallocation_scale`, `mrtpo/{correctness,format}_weight_std` (spread of
the token weights), `mrtpo/{correctness,format}_replay_fraction` (share of responses whose group varies on that reward)
and `mrtpo/replay_sequences` (sequences forwarded for the replay in the step, plus `mrtpo/replay_padding_sequences`,
one-row passes that keep the data-parallel ranks in step).
