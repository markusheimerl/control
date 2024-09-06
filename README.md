### Machine Learning Approach to Quadcopter Control

The quadcopter control problem is framed as a **Markov Decision Process (MDP)**, where the goal is to learn an optimal policy that maximizes long-term rewards using **Reinforcement Learning (RL)**. Below are the key components and equations that form the foundation of this approach:

---

#### **MDP Components**:
1. **States**: $`\mathcal{S}`$ 
2. **Actions**: $`\mathcal{A}`$ 
3. **Reward function**: $`\mathcal{R}`$ 
4. **State transition probabilities**: $`p(s', r|s, a)`$, representing the probability of transitioning to state $`s'`$ and receiving reward $`r`$ given current state $`s`$ and action $`a`$.

   $`p(s', r|s, a) = \text{Pr}(S_t = s', R_t = r | S_{t-1} = s, A_{t-1} = a)`$

---

#### **Returns and Discounted Returns**:
The objective is to maximize the expected return. The return can be defined as:

- **Undiscounted return**:
  $`G_t = R_{t+1} + R_{t+2} + \cdots + R_T`$

- **Discounted return** (more common):
  $`G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}`$

where $`\gamma \in [0,1]`$ is the discount factor that reduces the value of future rewards.

---

#### **Policies and Value Functions**:
A **policy** $`\pi`$ defines the agent's strategy, mapping states to a probability distribution over actions.

- **State-value function** $`v_\pi(s)`$: the expected return when starting from state $`s`$ and following policy $`\pi`$:

  $`v_\pi(s) = \mathbb{E}_\pi[G_t | S_t = s] = \mathbb{E}_\pi[\sum_{k=0}^{\infty} \gamma^k R_{t+k+1} | S_t = s]`$

- **Action-value function** $`q_\pi(s, a)`$: the expected return from state $`s`$, taking action $`a`$, and then following policy $`\pi`$:

  $`q_\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]`$

---

#### **Bellman Equation**:
The **Bellman equation** gives a recursive definition of the value function:

- **State-value Bellman equation**:
  $`v_\pi(s) = \sum_a \pi(a|s) \sum_{s',r} p(s', r | s, a) [r + \gamma v_\pi(s')]`$

This equation is key to solving for the value function under a given policy.

---

#### **Optimal Policies and Value Functions**:
The goal of RL is to find an **optimal policy** $`\pi_*`$ that maximizes the expected return. The corresponding **optimal state-value function** $`v_*`$ and **optimal action-value function** $`q_*`$ are:

- Optimal state-value function:
  $`v_*(s) = \max_\pi v_\pi(s)`$

- Optimal action-value function:
  $`q_*(s,a) = \max_\pi q_\pi(s,a)`$

Using the **Bellman optimality equation**:

- For state-values:
  $`v_*(s) = \max_{a} \sum_{s', r} p(s', r|s,a)[r + \gamma v_*(s')]`$

- For action-values:
  $`q_*(s,a) = \sum_{s', r} p(s', r|s,a)[r + \gamma \max_{a'} q_*(s', a')]`$

---

#### **Advantage Function**:
The **advantage function** $`A_t`$ measures how much better it is to take a specific action $`a`$ in state $`s`$ compared to the average action:

$`A_t = q_\pi(s, a) - v_\pi(s)`$

---

#### **Proximal Policy Optimization (PPO)**:
PPO is a popular RL algorithm that approximates the optimal policy by maximizing a **clipped surrogate objective function**:

$`L(\theta) = \mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)A_t)]`$

where:
$`r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_\text{old}}(a_t|s_t)}`$

PPO performs updates to both the policy network and value function while keeping updates constrained to avoid instability.

---

### Conclusion:
By modeling quadcopter control as an MDP and leveraging reinforcement learning through **PPO**, the Bellman equations are approximated to derive near-optimal control policies. This approach is particularly effective in handling high-dimensional environments and continuous action spaces without needing explicit knowledge of the environment's dynamics.
