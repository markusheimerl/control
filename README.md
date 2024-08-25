# Machine Learning Approach to Quadcopter Control

## Modeling the Control Problem as a Markov Decision Process

In a finite Markov Decision Process (MDP), we have finite sets of states $\mathcal{S}$, actions $\mathcal{A}$, and rewards $\mathcal{R}$. The key idea is that the next state and reward depend only on the current state and action:

$$p(s',r|s,a) = \text{Pr}\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a\}$$

This function $p$ defines the dynamics of the MDP.

## Returns and Episodes

We aim to maximize the expected return, defined as a function of the reward sequence. The simplest return is the sum of rewards:

$$ G_t = R_{t+1} + R_{t+2} + R_{t+3} + \cdots + R_T $$

More generally, we use a discounted return:

$$ G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} $$

Here, $\gamma \in [0,1]$ is the discount factor, making future rewards less valuable.

## Policies and Value Functions

A policy $\pi$ maps states to probabilities of selecting each action. The value function $v_\pi(s)$ for a policy $\pi$ is the expected return when starting in state $s$ and following $\pi$:

$$ v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi \left[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \middle| S_t = s\right] $$

Similarly, the action-value function $q_\pi(s,a)$ is the expected return starting from state $s$, taking action $a$, then following policy $\pi$:

$$ q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a] $$

## The Bellman Equation

The Bellman equation relates the value of a state to the values of its successor states:

$$ v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s', r | s, a) [r + \gamma v_\pi(s')] $$

This equation expresses that the value of a state is the expected immediate reward plus the discounted value of the next state, averaged over all possible actions and outcomes.

## Optimal Policies and Optimal Value Functions

Solving a reinforcement learning task means, roughly, finding a policy that achieves a lot of reward over the long run. For finite MDPs, we can precisely define an optimal policy in the following way. Value functions define a partial ordering over policies. A policy $\pi$ is defined to be better than or equal to a policy $\pi'$ if its expected return is greater than or equal to that of $\pi'$ for all states. In other words, $\pi \geq \pi'$ if and only if $v_\pi(s) \geq v_{\pi'}(s)$ for all $s \in S$. There is always at least one policy that is better than or equal to all other policies. This is an optimal policy. Although there may be more than one, we denote all the optimal policies by $\pi_*$. They share the same state-value function, called the optimal state-value function, denoted $v_*$, and defined as

$v_*(s) \doteq \max_\pi v_\pi(s)$

for all $s \in S$.

Optimal policies also share the same optimal action-value function, denoted $q_*$, and defined as

$q_*(s,a) \doteq \max_\pi q_\pi(s,a)$

for all $s \in S$ and $a \in A(s)$. For the state-action pair $(s,a)$, this function gives the expected return for taking action $a$ in state $s$ and thereafter following an optimal policy. Thus, we can write $q_*$ in terms of $v_*$ as follows:

$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s, A_t=a]$

Because $v_*$ is the value function for a policy, it must satisfy the self-consistency condition given by the Bellman equation for state values. Because it is the optimal value function, however, $v_*$'s consistency condition can be written in a special form without reference to any specific policy. This is the Bellman equation for $v_*$, or the Bellman optimality equation. Intuitively, the Bellman optimality equation expresses the fact that the value of a state under an optimal policy must equal the expected return for the best action from that state:

$v_*(s) = \max_{a\in A(s)} q_{\pi_*}(s,a)$

$= \max_a \mathbb{E}_{\pi_*}[G_t | S_t=s, A_t=a]$

$= \max_a \mathbb{E}_{\pi_*}[R_{t+1} + \gamma G_{t+1} | S_t=s, A_t=a]$ (by (3.9))

$= \max_a \mathbb{E}[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s, A_t=a]$

$= \max_a \sum_{s',r} p(s', r|s,a)[r + \gamma v_*(s')]$.

The last two equations are two forms of the Bellman optimality equation for $v_*$. The Bellman optimality equation for $q_*$ is

$q_*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1}, a') | S_t = s, A_t = a]$

$= \sum_{s',r} p(s', r|s,a)[r + \gamma \max_{a'} q_*(s', a')]$.

For finite MDPs, the Bellman optimality equation for $v_*$ has a unique solution. It's a system of equations, one for each state. With known environment dynamics, this system can be solved for $v_*$ using nonlinear equation solving methods.

Once $v_*$ is known, determining an optimal policy is straightforward:
- For each state, choose actions that maximize the right-hand side of the Bellman optimality equation.
- Any policy assigning nonzero probability only to these actions is optimal.
- A policy that is greedy with respect to $v_*$ is optimal.

The term "greedy" in this context refers to selecting alternatives based solely on immediate considerations. With $v_*$, a greedy policy becomes optimal in the long-term sense, as $v_*$ already accounts for future rewards.

Using $q_*$ makes optimal action selection even simpler:
- The agent can directly choose the action that maximizes $q_*(s,a)$ for any state.
- This eliminates the need for a one-step-ahead search.
- $q_*$ provides the optimal expected long-term return for each state-action pair.

By representing a function of state-action pairs, $q_*$ allows optimal action selection without knowledge of successor states, their values, or environment dynamics.

Explicitly solving the Bellman optimality equation to find an optimal policy is rarely practical:

1. It requires accurate knowledge of environment dynamics.
2. It demands extensive computational resources.
3. It assumes the Markov property holds.

These conditions are seldom met in practice. For example, backgammon has about 10^20 states, making direct solution computationally infeasible.

Alternative approaches:

1. Heuristic search methods: Expand the Bellman equation to a certain depth, using heuristic evaluation at leaf nodes.
2. Dynamic programming: Closely related to the Bellman equation.
3. Reinforcement learning methods: Approximately solve the Bellman equation using actual experienced transitions instead of expected ones.

These methods are explored in subsequent chapters as ways to approximately solve the Bellman optimality equation.


## Optimality and Approximation

Key points:
1. Optimal policies are rarely achievable in practice due to computational constraints.
2. The concept of optimality organizes the approach to learning and helps understand theoretical properties of algorithms.
3. Agents can only approximate optimal policies to varying degrees.

Challenges:
- Computational power: Even for well-defined environments, computing optimal policies is often infeasible.
- Memory constraints: Large memory is needed for approximating value functions, policies, and models.

Approximation methods:
- Tabular methods: Suitable for small, finite state sets.
- Function approximation: Necessary for large state spaces where tabular methods are impractical.

Benefits of approximation:
- Allows focus on frequently encountered states.
- Can achieve good performance even with suboptimal decisions in rare situations.

Example: TD-Gammon plays backgammon exceptionally well despite potentially making poor decisions in uncommon board configurations.

This approach distinguishes reinforcement learning from other methods of solving MDPs.