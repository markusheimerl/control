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