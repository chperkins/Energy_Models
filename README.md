# Energy_Models
Public website for showing code for thesis

This GitHub repository is a public space for sharing code for my Univesity of Chicago statistics master's thesis. In my thesis, I'm studying the best training practices for energy based models. Based on experimentation and literature review, the models are capable of very quickly producing realistic images based on MNIST or CIFAR10. However, they are also extremely sensitive to hyperparameters. This includes hyperparameters that cover the optimization of the the energy model, those that cover the minimization of the energy of the synethesized samples (traditionally done with Langevin dynamics), and additional regularization parameters. My goal is to provide a comprehensive analysis of what combinations of parameters work and why. Part of this research, both in theory and code, borrows heavily from "On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models" (https://github.com/point0bar1/ebm-anatomy) as well as "Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model" (https://github.com/enijkamp/short_run). Additionally, comparisons are made agasinst the coopnets model "Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching" (http://www.stat.ucla.edu/~ywu/CoopNets/main.html).

The results largely support these earlier papers but hope to delve further into the practical choices for each combination of hyperparameters and why they work. In particular, I currently find that while regularization, particularly in the form of spectral normalization, isn't necessary, it can be helpful when other parameters are set to speed up convergence. Similarly, traditional Langevin dynamics are completely avoided in the synthesized sample energy minimization. Instead, SGD is used with/without noise for similar effectiveness. 

This code is primarily updated privately in Google Colab for its computing resources. It will be updated here occasionally.

On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models
Erik Nijkamp*, Mitch Hill*, Tian Han, Song-Chun Zhu, and Ying Nian Wu (*equal contributions)
https://arxiv.org/abs/1903.12370
AAAI 2020.

Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model},
Nijkamp, Erik and Hill, Mitch and Zhu, Song-Chun and Wu, Ying Nian
NeurIPS 2019.

Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching
Xie, Jianwen and Lu, Yang and Gao, Ruiqi and Wu, Ying Nian
The 32nd AAAI Conference on Artitifical Intelligence 2018.
