

## Probability Path Formulation of Diffusion Models


- Score Matching Langevin Dynamics (SMLD): Variance Exploding (VE)
- Denoising diffusion probabilistic models (DDPM): Variance preserving
- sub-VP SDE (SM as SDE paper, Song)
- CondOT path (flow matching)
    - rectified flows
        - similar to CondOT, though not sure if there is some additional improvements
- others?




Really there's 3 main "formulations" of building a diffusion model:
1. score matching & Langevin dynamics
2. diffusion probabilistic models
3. Flow matching and CNF


There's also some work to tie these together, for instance DDPM (Ho et. al) shows that DDPM is equivalent to annealed score matching.

Song et. al show that re-formulating score matching as a continuous time problem they are able to show some equivalence betweeen score matching and CNF to maximize likelihood, and do likelihood calculations.




Stepping back more broadly there's 3 classes of generative models (according to Song):
1. implicit generative models (e.g. GAN)
2. likelihood based models (normalizing flows, VAE, EBM)
3. score based models


Score based models learn the gradient of log likelihood

$$
s_\theta(x) \sim \nabla_x \log p_(x)
$$




Score matching doesn't require computing the normalizing constant.

Consider
$$p (x) := \frac{g(x)}{Z}$$
Then
$$
\begin{aligned}
\nabla \log p(x) &= \frac{\nabla p(x)}{p(x)} \\
&= \frac{\nabla g(x)}{Z} \frac{Z}{g(x)} \\
&= \frac{\nabla g(x)}{g(x)}
\end{aligned}
$$

So we can just train some network $s(x)$ to approximate this function via Fisher divergence
$$
\mathbb{E}_{x \sim p(x)} \lbrack \| \nabla \log p(x) - s_\theta(x) \|^2_2 \rbrack
$$



conditional score matching ~ denoising score matching

For score matching 2 viable approaches in practice
- sliced score matching
    - requires forward mode AD though
- denoising score matching
    - add noise to calculate the score
    $$
    \mathbb{E}_{q(\tilde{x}|x), p_\text{data}(x)} \lbrack \|s_\theta(x) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) \|_2^2 \rbrack
    $$



#### Side note on MALA (Metropolis Adjusted Langevin Algorithm)
Technically we should be sampling based on MH to determine acceptance of the new sample, in order to ensure that our Markov chain converges to the correct distribution independent of our noise scale.

Proposal process for $x_{t+1}$ is based on Langevin sampling SDE (discretized of course, e.g. Euler-Maruyama):
$$
dX_t = \lbrack \frac{\epsilon^2}{2} \nabla \pi(x) \rbrack dt + \epsilon dW_t
$$

However since we're not able to calculate the unnormalized probability $\tilde{\pi}(x)$ we're not able to apply MALA, which is a different setting than typical Langevin dynamics.
It's useful to understand the difference in how Langevin dynamics is applied for SGM vs typical Bayesian inference using Langevin dynamics (where we typically assume some model which allows us to calculate $\tilde{\pi}(x)$.



#### Differences in approach of Diffusion Probabilistic Models (DPM) vs Score-based Generative Models (SGM)

DPM approaches the problem using hierarchical latent variables and a variational decoder (reverse process).

SGM approaches this problem from the score matching perspective i.e. how do we sample $p_\text{data}(x)$.



## References
- [Score-Based Diffusion Models](https://fanpu.io/blog/2023/score-based-diffusion-models/)
- [Generative Modeling by Estimating Gradients of the Data Distribution](https://yang-song.net/blog/2021/score/)
- [A Simplified Overview of Langevin Dynamics](https://friedmanroy.github.io/blog/2022/Langevin/)
