# FrozenLake - CS234 Assignment1
This project is about making an agent to control the movement of a character in a grid world. 
The gridworld is in form of a frozen lake where some tiles of the grid are walkable, and others lead to the agent falling into the water. 
Additionally, the movement direction of the agent is uncertain and only partially depends on the chosen direction. 
The agent is rewarded for finding a walkable path to a goal tile.

![4x4 gridworld](https://github.com/Tanmay-Pathrabe/FrozenLake-Assignment1/blob/main/Output%20clips/Frozen-Lake.png)

* *S: starting point, safe*
* *F: frozen surface, safe*
* *H: hole, fall in the hole*
* *G: goal, reward*

## Concept Used-
* __Policy iteration__ : Policy Iteration is an algorithm in ‘ReInforcement Learning’, which helps in learning the optimal policy which maximizes the long term discounted reward. 
n policy iteration algorithms, you start with a random policy, then find the value function of that policy (policy evaluation step), then find a new (improved) policy based on the previous value function, and so on. 
In this process, each policy is guaranteed to be a strict improvement over the previous one (unless it is already optimal).
***Policy evaluation + Policy improvement***, and the two are repeated iteratively until policy converges.


* __Value iteration__ : Value iteration is a method of computing an optimal MDP policy and its value. In value iteration, you start with a random value function and then find a new (improved) value function in an iterative process, until reaching the optimal value function
.Value iteration starts at the "end" and then works backward, refining an estimate of either Q* or V*. ***Finding value function + new policy extraction and improving value function*** and iterating over it again and again.

To read more, Click on the [link](https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf)

## Built with- 
* Pyhton3
* Jupyter Notebook

## Libraries used-
1. [numpy](https://numpy.org/doc/)
1. [gym](https://gym.openai.com/)
1. [time](https://docs.python.org/3/library/time.html)

## Platform-
* **Ubuntu 20.04**
