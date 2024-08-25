# control
A ML approach to quadcopter control

## Modeling the control problem as a markov decision process

In a finite MDP, the sets of states, actions, and rewards ($\mathcal{S}$, $\mathcal{A}$, and $\mathcal{R}$) all have a finite number of elements. In this case, the random variables $R_t$ and $S_t$ have well defined discrete probability distributions dependent only on the preceding state and action. That is, for particular values of these random variables, $s' \in \mathcal{S}$ and $r \in \mathcal{R}$, there is a probability of those values occurring at time $t$, given particular values of the preceding state and action:
$$p(s',r|s,a) \doteq \text{Pr}\{S_t=s', R_t=r | S_{t-1}=s, A_{t-1}=a\},$$
for all $s',s \in \mathcal{S}$, $r \in \mathcal{R}$, and $a \in \mathcal{A}(s)$. The function $p$ defines the dynamics of the MDP.