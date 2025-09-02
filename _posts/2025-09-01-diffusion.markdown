---
layout: post
title:  "Diffusion Models"
date:   2025-09-01 20:00:00 -0700
categories: diffusion
mathjax: true
---

# Diffusion Models


## Diffusion Model Cheatsheet
Recently I've been studying diffusion models, which are used in many SotA generative models (Stable Diffusion 3, Moviegen, Flux, Wan, etc.). The main resource I'm using is the MIT course [Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/), along with reading some key papers.

This post has 2 parts:
1. Flow and score matching model cheat-sheet based on mainly on the course material.
    - This is more like a reference to help remember key points *after* understanding the course material.
2. Why are diffusion models interesting? (opinions)


## Cheat-sheet
The goal of flow matching or score matching (and generative modeling in a general sense) is to sample from the data distribution. For flow matching (and score matching) the way to do this is to map from an initial distribution $p_\text{init}$ to the data distribution $p_\text{data}$. The way to do this is to construct a probability path $p_t$ for $t \in \lbrack 0, 1 \rbrack$, and then train a model that will allow us to sample from this distribution at $t = 1$.



There are many different formulations for diffusion model. Usually when the term 'diffusion model' is used, what is meant is a denoising diffusion model, similar to what is described in [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239). In the case of diffusion models, the initial distribution and probability paths are always Gaussian.


$X_t$ is a (random, in the case where $x_0 \sim p_\text{data}$) trajectory, $u_t$ is the vector field defining the ODE, and $\psi_t$ is the flow function mapping from initial condition and time to the state at time $t$.

$$
\begin{aligned}
X_t &= \psi_t(x_0) \\
X_0 &= x_0 \\
X_0 &= \psi_0(x_0) \\
\frac{dX_t}{dt} &= u_t(X_t) \\ 
\frac{d}{dt} \psi_t(x_0) &= u_t(X_t) = \psi_t(\psi_t(x_0))
\end{aligned}
$$

If $u_t$ Lipschitz then $\psi_t$ is a diffeomorphism and is the unique solution to the ODE 

$$\frac{d}{dt}X_t = u_t(X_t)$$

In addition to the probability path $p_t$, we define a conditional probability path conditioned on $z \sim p_\text{data}$ $p_t(x | z)$. Such a probability path interpolates between our $p_\text{init}$ at $t = 0$ and $\delta_z$ at $t = 1$.
Note when we marginalize this distribution at $t = 0, 1$ respectively we get $p_\text{init}$ and $p_\text{data}$, our target starting and ending distributions. Defining the conditional distribution is helpful, as our training loss will be the **conditional** flow matching loss.

There are many probability paths that would suffice for interpolating between the dirac delta and initial distribution (depending on what we choose the initial distribution). As previously mentioned, in diffusion models the choice is restricted to Gaussian probability paths, and $p_\text{init}$ can be assumed to be the standard normal distribution.

In the case of Gaussian probability path, we define the conditional probability path as

$$
\begin{aligned}
p_t(x | z) &= \mathcal{N}(\alpha_t z, \beta_t^2) \\
x &= a_t z + \beta_t \epsilon \qquad s.t. \quad \epsilon \sim N(\mathbb{0}, I_d) \\
\alpha_0 &= \beta_1 = 0 \\
\alpha_1 &= \beta_0 = 1
\end{aligned}
$$ 


which satisfies the interpolation requirement.
$\alpha_t$ and $\beta_t$ parameters are usually referred to as noise schedule parameters, with the additonal requirement that they be continuous and monotonic.

From the conditional probability path, we are able to have a closed form solution for conditional vector field in terms of our noise schedule

$$
u_t(x|z) = (\dot{\alpha_t} - \frac{\dot{\beta_t}}{\beta_t} \alpha_t) z + \frac{\dot{\beta_t}}{\beta_t}x
$$

Next marginalize the conditional vector field:

$$
u_t^\text{target}(x) = \int u_t^\text{target}(x | z) p_\text{data}(z|x) dz
$$


Apply the continuity equation to show that the corresponding ODE defined by the marginal vector field follows the marginal probability path.

The training loss will be


$$
\mathcal{L}_\text{FM} = \mathbb{E}_{t \sim U[0, 1], x \sim p_t} \lbrack \Vert u_t^\theta(x) - u_t^\text{target}(x) \Vert^2 \rbrack
$$


The problem is that it is not possible to sample $x$ directly and $u_t^\text{target}(x)$ is intractable.
However, it can be show that


$$
\begin{aligned}
&\mathcal{L}_\text{CFM} := \mathbb{E}_{t \sim U[0,1], z \sim p_\text{data}} \lbrack \Vert u_t^\theta(x|z) - u_t^\text{target}(x|z) \Vert^2 \rbrack \\ 
&\mathcal{L}_\text{FM} = \mathcal{L}_\text{CFM} + C
\end{aligned}
$$

For some $C$ that doesn't depend on $\theta$.

Then $\nabla \mathcal{L}_\text{FM} = \nabla \mathcal{L} _ \text{CFM}$ and we can consider this the same minimization problem as the marginal case.

Once we have a learned $u_t^\theta$, inference is straightforward.
Sample $x_0 \sim p_\text{init}$, and solve the ODE.


Let's back up to take a look at the score matching case.
In the course this is handled as an SDE extension to the ODE corresponding to the vector field.

$$
\begin{aligned}
& X_0 \sim p_\text{init}; \\
& dX_t = \lbrack u_t^\text{target}(X_t) + \frac{\sigma_t^2}{2} \nabla \log p_t(X_t) \rbrack dt + \sigma_t dW_t
\end{aligned}
$$

This SDE extension is valid for any $\sigma_t \geq 0$. My rough intuition is that we can add more noise (the Brownian motion term), but to compensate we increase also the weighting of the score function $\nabla \log p_t(X_t)$ which keeps the probability distribution close to the desired distribution.

We can use Fokker-Planck to show that the probability path corresponding to this SDE is identical to the probability path corresponding to the ODE defined by the flow field.


Similar to the flow field, we can't calculate the the marginal score function directly and instead we need to marginalize from the conditional score function:

$$
\nabla \log p_t(x) = \int \nabla \log p_t(x|z) p_t(z|x) dz
$$

For the Gaussian path case, the conditional score function is

$$
\nabla \log p_t(x | z) = - \frac{x - \alpha_t z}{\beta_t^2}
$$

During training, instead of minimizing the $L^2$ difference of marginal flow fields, we instead minimize the difference of marginal score functions. Similar to flow matching, the conditional and marginal losses are same up to a constant so the minimizer of the conditional loss (which can be calculated in closed form) is sufficient.

$$
\begin{aligned}
\mathcal{L}_{SM}(\theta) &= \mathbb{E}_{t \sim U[0,1], z \sim p_\text{data}, x \sim p_t(\cdot | z)} \lbrack \Vert s_t^\theta(x) - \nabla \log p_t(x) \Vert^2 \rbrack \\
\mathcal{L}_{CSM}(\theta) &= \mathbb{E}_{t \sim U[0,1], z \sim p_\text{data}, x \sim p_t(\cdot | z)} \lbrack \Vert s_t^\theta(x) - \nabla \log p_t(x|z) \Vert^2 \rbrack \\
\mathcal{L}_\text{SM}(\theta) &= \mathcal{L}_\text{CSM}(\theta) + C \\ 
\nabla \mathcal{L}_\text{SM} &= \mathcal{L}_\text{CSM}
\end{aligned}
$$

During sampling, we can just solve the above SDE for $dX_t$, for instance using Euler-Maruyama or similar numerical solver.

In the case of Gaussian probability paths there is an equivalence between score function and flow, meaning finding the minimizer of one is enough:

$$
u_t^\text{target}(x) = (\beta_t^2 \frac{\dot{\alpha_t}}{\alpha_t} - \dot{\beta_t}\beta_t) \nabla \log p_t(x) + \frac{\dot{\alpha_t}}{\alpha_t} x
$$

In practice there may be differences to solving one minimization problem over another, as $s_t^\theta$ and $u_t^\theta$ are just approximations of the score function and flow field. Similarly, when doing inference there may be some advantages to using flow matching over score matching, or differences in the choice of $\sigma_t$ for simulating the SDE.


## Interesting Connections

### Score matching <-> Flow Matching
The most salient point I remember when I studied SDE was the connection between SDE (stochastic) and PDE (e.g. diffusion), and here that equivalence can be seen applied. The probability path in this case is the PDE, and the SDE of course is the $dX_t$ term above. Although the course approaches teaching the SDE approach (score matching) as the extension, it's good to remember that the early formulation of diffusion models is closer to the discretized SDE than the flow matching case. Flow matching is conceptually cleaner, and more robust so in pedagogy makes sense to introduce first. I'm still curious though, when does it make sense to choose to simulate the SDE vs the ODE in practice?


### Langevin Dynamics <-> Score Matching
The special case where $p_t = p$ is held static is the case of Langevin dynamics.
One way to sample from $p$ when one is able to calculate the score function is to sample from this SDE using MCMC

$$
dX_t = \frac{\sigma_t^2}{2} \nabla \log p(X_t) dt + \sigma_t dW_t
$$

I wonder if it makes sense to freeze the flow term at some point to sample from some stationary distribution.


### Continuous Normalizing Flows <-> Flow Matching
One of my favorite papers from the late 201Xs is [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366), which introduces a way to parameterize the solution to ODEs using some learned DNNs. The key contribution is to realize that neural networks can be used to approximate vector fields, which can be used to numerically simualte ODE solutions. Backpropagation can be done via adjoint sensitivity, where an augmented ODE is solved in reverse time to get gradients w.r.t some hidden state, as well as initial conditions and starting and stopping times of the ODE.

One of the applications for NODE presented in the paper is Continuous Normalizing Flows (CNF), where normalizing flows (basically a discrete, learned reversible map between data and latent space) is made continuous. It turns out that training CNF (and NODE in general) is quite different due to needing to also run an ODE solver during training. However, the same idea of having a neural network parameterize the vector field of a probability flow is the same idea in flow matching, even if the training objective is different (in CNF it's maximum likelihood, in CNF it's approximating the flow vector field).


## References
- [Introduction to Flow Matching and Diffusion Models](https://diffusion.csail.mit.edu/)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456)
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747)
- [Neural Ordinary Differential Equations](https://arxiv.org/abs/1806.07366)
