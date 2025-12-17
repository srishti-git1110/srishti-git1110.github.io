---
title: Reinforcement Learning
layout: post
date: 2025-06-10
tags:
    - Reinforcement Learning, Decision Making
description: 
draft: true
---

My first version of this blog read like a boring text book; it was highly technical but lacked the purpose of being helpful to anyone willing to start learning RL.

The version that I'm publishing now is aimed at people willing to start. That said, the first read could still be intimidating because I also didn't want to lose technical rigour. 

# Prep: Formalizing the RL problem
Reinforcement learning is the science of decision making. What does it mean to make decisions? Making decisions requires one to take into account a couple of factors and then take an action(s); the actions are chosen such that one is able to obtain the best reward from them in the long term. RL is no different.

I think it's already nice to note that we're talking about sequential decision making where the decisions taken today are relevant to the long term outcomes.

The way humans "learn" to make decisions using their life experiences is analogous to the "learning" part in Reinforcement Learning.

Bear with me for this analogy before we jump right into the foundational jargons needed to formally define the RL setup. 

### Analogy

As humans, we live in an environment/world and we ought to make decisions at every point in our lives. If I state the obvious more explicity, I'd go: As humans, we need to make decisions all the time keeping in mind a few factors—the past until the present point in time, the present state of our lives, the immediate state our decision will land us in, both the short term gain received in response to our decision and also our long term goals, and so on. All of this is difficult because of one single fact—Uncertainty. Of what? Of the future. The future is uncertain and hence it's hard to know in advance the actual outcome or the response from the world received by our actions. 

This is really important. If this uncertainty wasn't there, the science of decision making would've been much more simpler. At this point, it can also benefit to concretely define the two sources of uncertainty—the exact immediate/next state our current action/decision would land us in, and the gain that our current decision would yield immediately. 

For example, consider an individual who currently needs to decide between accepting an admit to a graduate program and accepting a high paying job at a newly found startup. Let's suppose their long term goal is to also build a startup of their own so they can earn a lot of $$. Given that, their current decision needs to be the one that helps fulfill that goal in the most optimized manner. Hence, the two reasons why it's hard to make this decision are:

- The individual is uncertain of the benefit yielded by both decisions. Maybe the MS program yields more benefit via providing a strong knowledge base that could be leveraged to build a startup of maybe the job yields better benefit by providing strong hands-on exp!

- The individual is also uncertain of the state both decisions land them in. Maybe the job environment as a fast paced startup quickly puts them in more senior or managerial positions that can be better for their own startup or maybe the MS programme would allow them to meet like-minded peers that could be their co-founders.

*And of course, this was my way of characterizing the problem. One could obviously define the benefit received and the next state differently.*

 

### The Jargon

As noted, RL is the science of sequential decision making under uncertainty with a goal to optimize (of course, optimize here = maximize) the long term rewards. 

In order to mathematically formalize and solve a problem as an RL problem, we need to define the following terms:
- The *environment* the agent is interacting with.
- An *agent*. The one that interacts with the environment, takes decisions and acts on them.
- The *state* of the agent at any time step t, st 𝜖 S where S is the state space defining all possible states the agent can be in. The state can be thought of any information that's useful in deciding what action to take.
- The *action(s)* taken by the agent at any time step t, at 𝜖 A where A is the action space defining all possible actions that can be taken by the agent. 
- The *policy* followed by the agent according to which it acts.




![the rl setting](agent.png#center)

I'll now elaborate on some of these terms along with their mathematical equations and that should make the full RL setting clear.

**Policy.** We say that our agent takes actions. Now, how does the agent know which action to take in which state? The Policy defines that. So think of the policy as being analogous to the humain brain. While we're at it, realise how our brain "learns" to make decisions. And hence, the whole field of RL is concerned with "learning" a good policy for the agent. We say, xyz RL algorithm—what is the algorithm doing? Just finding a good policy for the agent.



Mathematically, the policy is just the conditional probability distribution of s 𝜖 S given a 𝜖 A.

👉 *I said 'good' policy. Think about what a good policy is. We've only loosely touched it until here and will more formally study it further.*

<exp>




<style>
.collapsible {
  background-color: #090909ff;
  cursor: pointer;
  padding: 10px;
  width: 100%;
  border: none;
  text-align: left;
  outline: none;
  font-size: 1em;
}

.active, .collapsible:hover {
  background-color: #ddd;
}

.content {
  display: none;
  padding: 10px 0;
  border-top: 1px solid #ccc;
  text-align: center
}
</style>

<button class="collapsible">▶ The policy can either be stochastic or deterministic.</button>
<div class="content">
  <strong>Stochastic Policy.</strong> Given a state s, a probability distribution can be defined over the action space A. <br>
  enter expression here. <br>
  i.e. In any state, the agent can take one or more actions depending on the probability of taking each as determined by the policy. <br>
  <strong>Deterministic Policy.</strong> Given a state s, the agent will take only one action defined by the policy. <br>
  enter exp <br><br><br>
  Being verbose: what do I mean by "the policy can be stochastic or deterministic? I mean that based on the domain we're using RL in, we can choose to *model* it either way—either mathematical expression can be used and then we learn it.



  
</div>

<script>
document.querySelectorAll(".collapsible").forEach(function(btn) {
  btn.addEventListener("click", function() {
    this.classList.toggle("active");
    var content = this.nextElementSibling;
    if (content.style.display === "block") {
      content.style.display = "none";
      this.textContent = "▶ Click to show explanation";
    } else {
      content.style.display = "block";
      this.textContent = "▼ Hide explanation";
    }
  });
});
</script>


**Environment.** When we say the agent interacts with the environment, we mean two things:
- The Reward function: The environment gives some reward Rt to the agent when it takes some action At in state St.
Mathematically, the Reward Function R can be written as:
R (s, a) = E[Rt+1 | St=s, At=a]
i.e. the expected reward from taking action a while being in state s.

The important part to clear here is the reason why we need to define the Reward Function as an expectation. The reason is that upon taking action a in state s, the agent can transition to one or more possible states s' and the reward received in each case is different. And because R is only a function of the current state s and the current action a, we need to average it over the set of next possible states the agent can transition to.

- The Dynamics Model (Transition probabilities): The environment also determines the next state St+1 the agent transitions to upon taking action At in state St.

exp

**Horizon.** An episode is one trajectory of actions taken by the agent.
Horizon, simply, is the number of time steps until we consider an episode to last. It could be infinite.

**Return.** *For any sample episode*, the Return at time step t is defined as the discounted sum of rewards from t till the horizon.
Hence, the return is simply:

Gt = Rt + gamma Rt+1 + ...

Note that here we use the term Rt and not R(s, a) (the reward function) and that's clearly because we're just noting the rewards received by the agent in any sample episode.

Why the discount factor gamma?

Gamma makes things simpler mathematically. Eg. if the episode continues for a very large number of steps or for infinite number of steps , gamma makes it easier to calculate the Return.
It's also used to weigh down the importance of future rewards as needed—if the future rewards matter as much as the immediate reward, gamma can be set to 1; and can otherwise be lowered as needed.

**Value Function.** 

**Q Function.**

**Goal of an RL problem - Optimal policy.** As already discussed, our goal for the agent is to find a good/optimal policy. We've also discussed that the agent ought to take actions such that the long term rewards are maximized.

Putting it all together, an optimal policy is the one corresponding to which the value functions attains its highest value (and of course, the value function is defined per state and that's how we obtain the policy which is just the optimal actions to be taken per state).

The optimal value/Q function, , is just the value/Q function corresponding to the optimal policy.



# The Markovian Property

# Markov Reward Process (MRP)

# Markov Decision Process (MDP)

# Policy Evaluation and Policy Search (Iteration)

# Value Iteration

# Monte Carlo Methods

# Temporal Difference Learning