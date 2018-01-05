<img src="data/logo.jpg" width=25% align="right" />

# This fork difference from OpenAI Baselines

Currently this fork differs from the OpenAI master branch after [commit 8c90f67](https://github.com/openai/baselines/commit/8c90f67560920224e466d0389ac1bbf46e00773c).  The baselines/deepq/experiments/atari module has enjoy.py and train.py which are both broken by an earlier [commit bb40378](https://github.com/openai/baselines/commit/bb403781182c6e31d3bf5de16f42b0cb0d8421f7).  These files relied on metadata added to the info object by SimpleMonitor, and the commit removes SimpleMonitor and replaces to call to it with another monitor which doesn't add the necessary info fields.  As well, there is some deprecated module removed which was still referenced by these files.

This fork restores SimpleMonitor, adds it to the env, and changes the wrapper calls to current code (away from the deprecated/removed module).

# Baselines

OpenAI Baselines is a set of high-quality implementations of reinforcement learning algorithms.

These algorithms will make it easier for the research community to replicate, refine, and identify new ideas, and will create good baselines to build research on top of. Our DQN implementation and its variants are roughly on par with the scores in published papers. We expect they will be used as a base around which new ideas can be added, and as a tool for comparing a new approach against existing ones. 

You can install it by typing:

```bash
git clone https://github.com/openai/baselines.git
cd baselines
pip install -e .
```

- [A2C](baselines/a2c)
- [ACER](baselines/acer)
- [ACKTR](baselines/acktr)
- [DDPG](baselines/ddpg)
- [DQN](baselines/deepq)
- [PPO1](baselines/ppo1) (Multi-CPU using MPI)
- [PPO2](baselines/ppo2) (Optimized for GPU)
- [TRPO](baselines/trpo_mpi)

To cite this repository in publications:

    @misc{baselines,
      author = {Dhariwal, Prafulla and Hesse, Christopher and Plappert, Matthias and Radford, Alec and Schulman, John and Sidor, Szymon and Wu, Yuhuai},
      title = {OpenAI Baselines},
      year = {2017},
      publisher = {GitHub},
      journal = {GitHub repository},
      howpublished = {\url{https://github.com/openai/baselines}},
    }
