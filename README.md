# Energy_Models
Public website for showing code for thesis

This GitHub repository is a public space for sharing code for my University of Chicago statistics master's thesis. In my thesis, I'm studying the best training practices for energy based models. The energy based model training uses the following process. A model (usually a neural net) takes in an input and outputs a scalar value. There is no restriction on this value. It can be positive or negative and is unbounded. In theory, this represents the energy of the input. There is a connection between the energy and the log probability density of the image. We want a model that provides a large scalar for images that are representative of our dataset and small values for other inputs. 

To train energy based models, we use the gradient of the difference between the mean energy of observed data and the mean energy with respect to the model (AKA synthesized data). To generate synthesized data, one usually initializes images in some way, either with noise, data, or something else. Then, these are run through MCMC to transform them into images with high values from the model AKA high probability density. In other words, we synthesize images that are representative of our current model. How one decides to synthesize images can heavily change the performance.

The gradient of the model is simple, and the models are capable of producing realistic images based on MNIST or CIFAR10. However, they are also extremely sensitive to hyperparameters. This includes hyperparameters that cover the optimization of the the network weights, those that cover the MCMC process for synthesizing images, and additional regularization parameters. My goal is to provide a comprehensive analysis of what combinations of parameters work and why. Part of this research, both in theory and code, borrows heavily from "On the Anatomy of MCMC-Based Maximum Likelihood Learning of Energy-Based Models" (https://github.com/point0bar1/ebm-anatomy) as well as "Learning Non-Convergent Non-Persistent Short-Run MCMC Toward Energy-Based Model" (https://github.com/enijkamp/short_run). Additionally, comparisons are made against the coopnets model "Cooperative Learning of Energy-Based Model and Latent Variable Model via MCMC Teaching" (http://www.stat.ucla.edu/~ywu/CoopNets/main.html).

The number of steps used for synthesizing data appears to be the most important, and all other hyperparameters need to be set around it. With a small number of steps, the overall learning rate of the model needs to be low, and added spectral normalization in layers of the network can further stabilize training. With a larget number of steps used, the model becomes more stable. However, it also takes much longer to train. It also appears that proper MCMC is not necessary. One can replace this with simple gradient ascent and get comparable results on these image datasets.

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
