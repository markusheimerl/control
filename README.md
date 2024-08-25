# control
A ML approach to quadcopter control

## Modeling the control problem as a markov decision process

In a finite MDP, the sets of states, actions, and rewards ($\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$) all have a finite number of elements. In this case, the random variables $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action. That is, for particular values of these random variables, $s' \in \mathcal{S}$ and $r \in \mathcal{R}$, there is a probability of those values occurring at time $t$, given particular values of the preceding state and action:
$$p(s',r|s,a) \doteq \text{Pr}\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a\},$$
for all $s',s \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A}(s)$. The function $p$ defines the dynamics of the MDP.

### Returns and Episodes

So far we have discussed the objective of learning informally. We have said that the agent's goal is to maximize the cumulative reward it receives in the long run. How might this be defined formally? If the sequence of rewards received after time step $t$ is denoted $R_{t+1}, R_{t+2}, R_{t+3}, \ldots$, then what precise aspect of this sequence do we wish to maximize?

In general, we seek to maximize the expected return, where the return, denoted $G_t$, is defined as some specific function of the reward sequence. In the simplest case the return is the sum of the rewards:

$$ G_t \doteq R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T, \qquad (3.7) $$

...

Thus,
we can define the return, in general, according to (3.8), using the convention of omitting
episode numbers when they are not needed, and including the possibility that γ = 1 if the
sum remains defined (e.g., because all episodes terminate). Alternatively, we can write

$$ G_t \doteq \sum_{k=t+1}^T \gamma^{k-t-1} R_k, \qquad (3.11) $$

including the possibility that T = ∞ or γ = 1 (but not both). We use these conventions
throughout the rest of the book to simplify notation and to express the close parallels
between episodic and continuing tasks.

...

### Policies and value functions

Formally, a policy is a mapping from states to probabilities of selecting each possible
action. If the agent is following policy π at time t, then π(a|s) is the probability that
A_t = a if S_t = s. Like p, π is an ordinary function; the "|" in the middle of π(a|s)
merely reminds that it defines a probability distribution over a ∈ A(s) for each s ∈ S.
Reinforcement learning methods specify how the agent's policy is changed as a result of
its experience.

The value function of a state s under a policy π, denoted v_π(s), is the expected return
when starting in s and following π thereafter. For MDPs, we can define v_π formally by

$$ v_π(s) \doteq \mathbb{E}_π[G_t | S_t = s] = \mathbb{E}_π \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s\right], \text{ for all } s \in \mathcal{S} $$

where $\mathbb{E}_π[\cdot]$ denotes the expected value of a random variable given that the agent follows
policy π, and t is any time step. Note that the value of the terminal state, if any, is
always zero. We call the function v_π the state-value function for policy π.

Similarly, we define the value of taking action a in state s under a policy π, denoted
q_π(s, a), as the expected return starting from s, taking the action a, and thereafter
following policy π:

$$ q_π(s, a) \doteq \mathbb{E}_π[G_t | S_t = s, A_t = a] = \mathbb{E}_π \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s, A_t = a\right] $$

We call q_π the action-value function for policy π.

### Policies and value functions

A fundamental property of value functions used throughout reinforcement learning and
dynamic programming is that they satisfy recursive relationships similar to that which
we have already established for the return (3.9). For any policy π and any state s, the
following consistency condition holds between the value of s and the value of its possible
successor states:

$$ v_π(s) \doteq \mathbb{E}_π[G_t | S_t = s] $$
$$ = \mathbb{E}_π[R_{t+1} + \gamma G_{t+1} | S_t = s] \qquad \text{(by (3.9))} $$
$$ = \sum_a \pi(a|s) \sum_{s'} \sum_r p(s', r | s, a) [r + \gamma \mathbb{E}_π[G_{t+1}|S_{t+1} = s']] $$
$$ = \sum_a \pi(a|s) \sum_{s',r} p(s', r | s, a) [r + \gamma v_π(s')], \quad \text{for all } s \in \mathcal{S}, \qquad (3.14) $$

where it is implicit that the actions, a, are taken from the set A(s), that the next states,
s', are taken from the set S (or from S+ in the case of an episodic problem), and that
the rewards, r, are taken from the set R. Note also how in the last equation we have
merged the two sums, one over all the values of s' and the other over all the values of r,
into one sum over all the possible values of both. We use this kind of merged sum often
to simplify formulas. Note how the final expression can be read easily as an expected
value. It is really a sum over all values of the three variables, a, s', and r. For each triple,
we compute its probability, π(a|s)p(s', r | s, a), weight the quantity in brackets by that
probability, then sum over all possibilities to get an expected value.