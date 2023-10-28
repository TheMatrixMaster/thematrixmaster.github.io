---
layout: distill
title: Variational Inference w/ EM algorithm (Part 1)
date: 2023-10-09
tags: bayesian variational inference statistics theory
giscus_comments: true

authors:
  - name: Stephen Lu
    url: "https://matrixmaster.me"
    affiliations:
      name: McGill University

bibliography: 2023-10-09-variational-inf.bib
---

## Motivation
Given an underlying distribution $$p_r$$ with unknown parameters $$\theta$$, the goal of Bayesian inference is to learn a posterior distribution over $$\theta$$ given a dataset $$X$$ whose examples are sampled independently from $$p_r$$. Given that $$p_r$$ might be a very complex distribution over many random variables, one trick to simplify the task involves introducing latent variables $$z$$ that break down the overall inference problem into smaller subproblems (like a **divide-and-conquer** approach). Given these additional latents, our objective is to infer the joint posterior distribution over the unknown parameters $$\theta$$ and latents $$z$$ given an observed dataset $$X$$ according to Bayes' rule.

$$
\begin{equation}
\label{eq:bayes}
p(\theta,z|X) = \frac{p(X\vert\theta,z)p(\theta|z)p(z)}{p(X)}
\end{equation}
$$

Although the latents simplify the problem, this posterior remains intractable given that we need to marginalize over all parameters and latents to compute the evidence $$p(X)=\int{\int{p(X,\theta,z)dz\,}d\theta}$$.

To address this, one can attempt to approximate the evidence using stochastic sampling (Monte Carlo methods), but this optimization procedure is not interpretable, is compute-intensive, and requires many samples for convergence. Another approach, known as mean-field variational inference (VI), allows us to completely bypass the evidence computation issue by making a few additional assumptions about $$\theta$$ and $$z$$.

In this (part 1) post, I will go over the theory behind this method, and in the next post (part 2), I'll walk through my implementation of mean-field VI on the task of polygenic risk score (PRS) regression with spike-and-slab prior.

### Variational Distribution by Mean-Field
Given that the evidence term is intractable, variational inference proposes that we learn a simpler distribution over the unknown parameters and latents $$q(\theta,z)$$ that approximates the true posterior $$p(\theta,z|X)$$. Most importantly, the main idea behind mean-field VI is that we can strategically restrict $$q(\theta,z)$$ to a simpler distribution family than $$p(\theta,z\vert X)$$, while optimizing the parameters of $$q$$ to obtain a good approximation of $$p$$. The only assumption we need for this to work is the mean-field approximation, i.e. conditional independence among some partition of the latents $$z$$ into $$z_1, ..., z_M$$ such that

$$
\begin{equation}
\label{eq:assumption}
q(\theta, z \vert X) = \prod_i q(\theta \vert z_i)q(z_i)
\end{equation}
$$

In the variational inference lingo, we call $$q$$ the variational distribution.

### Evidence Lower Bound (ELBO)
Now that we've defined a factorization of $$q$$ into a partition of latent variables, we need to actually find an explicit distribution family for $$q(\theta \vert z_i)$$ and $$q(z_i)$$ that has enough capacity to adequately approximate $$p$$. Intuitively, a good variational distribution $$q$$, parametrized by $$\phi$$, should minimize the KL divergence between itself and the target posterior $$p$$. Let's take a closer look at this KL divergence expression:

$$
\begin{equation}
\label{eq:elbo}
\begin{split}
\mathbb{KL}(q\,\|\, p) & = \int{\int{q(\theta,z) \log\frac{q(\theta,z)}{p(\theta,z)} dz}\, d\theta} \\
    & = \int{\int{q(\theta,z) \log\frac{q(\theta,z)p(X)}{p(X,\theta,z)} dz\,}d\theta} \\
    & = \int{\int{q(\theta,z) [\log\frac{q(\theta,z)}{p(X,\theta,z)} + \log{p(X)}] dz}\, d\theta} \\
    & = \int{\int{q(\theta,z) \log\frac{q(\theta,z)}{p(X,\theta,z)} dz}\, d\theta} + \log{p(X)}\int{\int{q(\theta,z)dz}\, d\theta} \\
    & = -\text{ELBO}(q,\phi) + \log{p(X)}
\end{split}
\end{equation}
$$

Given that $$p(X)$$ is a constant, it is easy to see that minimizing $$\mathbb{KL}(q\,\|\, p)$$ is **equivalent** to maximizing $$\text{ELBO}(q,\phi)$$. Further, since KL divergence is a non-negative term, the following inequality holds $$\text{ELBO}(q,\phi) \leq \log(p(X))$$ — hence the name **evidence lower-bound**.

### Variational Expectation Maximization
So, now that we've shown that maximizing $$\text{ELBO}(q,\phi)$$ with respect to $$\phi$$ yields a good variational distribution $$q(\theta,z)$$, the next step is to derive a closed form distribution family for $$q(\theta,z\vert\phi)$$ that we can feed into the ELBO maximization scheme with respect to $$\phi$$. To do this, let's examine the ELBO expression in more detail, while using the mean-field assumption from $$\eqref{eq:assumption}$$. For simplicity, from now on, I'll omit explicitly dealing with the $$\theta$$ parameter by including it in the partition of latents $$z_1, ..., z_M, \theta$$ as an additional partitioned independent variable.<d-cite key="choy2017"></d-cite>

$$
\begin{equation}
\label{eq:varelbo}
\begin{split}
\text{ELBO}(q,\phi) & = \int{q(z\vert\phi)\log{\frac{p(X,z)}{q(z\vert\phi)}}dz} \\
    & = \int \prod_i q(z_i\vert\phi) \log p(X,z) dz - \sum_i \int q(z_i\vert\phi) \log q(z_i\vert\phi) dz_i \\
    & = \sum_j \int q(z_j) (\int \prod_{i\neq j}q(z_i)\log p(X,z) \prod_{i\neq j}dz_i\, dz_j) \\
    & \qquad - \int q(z_j)\log q(z_j)dz_j - \sum_{i\neq j}\int q(z_i)\log q(z_i)dz_i \\
    & = \sum_j \int q(z_j)\log \frac{\text{exp}(\mathbb{E}_{q(z_i)}[\log p(X,z)]_{i\neq j})}{q(z_j)}dz_j - \sum_{i\neq j}\int q(z_i)\log q(z_i)dz_i \\
    & = -\mathbb{KL}[q_j \| \tilde{p}_{i\neq j}] + \mathbb{H}(z_{i\neq j}) + C
\end{split}
\end{equation}
$$

Here, we see that the mean field assumption of independence between latents $z$ allows us to actually factorize the ELBO term into the sum over a function of each latent variable $$z_j \in \{z_1, ..., z_M, \theta\}$$. Given that the entropy term in the above equation $$\mathbb{H}(z_{i\neq j})$$ is fixed for a specific choice of $$\phi$$, we conclude that the overall ELBO is maximized when the negative KL divergence term equals zero, i.e. $$-\mathbb{KL}[q_j \| \tilde{p}_{i\neq j}] = 0$$ for all latents $$j$$, which only occurs when

$$
\begin{equation}
\label{eq:objective}
\begin{split}
\log q(z_j\vert\phi) & = \log \tilde p_{i\neq j} \\
    & = \mathbb{E}_{q(z_i\vert\phi)}[\log p(X,z)]_{i\neq j} + C \\
    & = \frac{\mathbb{E}_{q(z_i\vert\phi)}[\log p(X,z)]_{i\neq j}}{\int \mathbb{E}_{q(z_i\vert\phi)}[\log p(X,z)]_{i\neq j}\, dz_j} \\
    & = \frac{1}{C'}\mathbb{E}_{q(z_i\vert\phi)}[\log p(X,z)]_{i\neq j}
\end{split}
\end{equation}
$$

It is important to understand that if $$\mathbb{E}_{q(z_i)}[\log p(X,z)]_{i\neq j}$$ is tractable, then $$C'$$ is also tractable given that we have a finite set of latent variables with size $$M$$. And indeed, this expression is usually efficient to compute for a **specific choice** of $$\phi$$ if we choose known distributions for the joint-likelihood $$p(X,z) = p(X\vert z)p(z)$$. The keyword here is **specific** since it remains intractable to directly solve the above equation with respect to $$\phi$$ for most complex (and interesting) models of $$p(X,z)$$.

Thus, an approach that naturally comes to mind then is to iteratively update $$\phi$$ using gradient ascent on the ELBO objective. This approach is known as expectation maximization, or more specifically in this case as coordinate ascent in mean-field variational inference. This algorithm is very simple and comprises two steps that we iteratively apply until the ELBO objective converges to a maximum.<d-cite key="li2023"></d-cite>

#### Initialization
Before we run the EM algorithm, we need to choose some initial value for the parameters $$\phi$$, such that we may actually start the gradient ascent somewhere.

#### Expectation (E-Step)
In the E-step, we simply *update* the probability distribution of our variational distribution by evaluating $$\eqref{eq:objective}$$ with the current version of $$\phi$$ as follows:

$$
\begin{equation}
\label{eq:e-step}
\begin{split}
& \log q(z_j)' = \frac{1}{C'}\mathbb{E}_{q(z_i\vert\phi)}[\log p(X,z\vert \phi)]_{i\neq j} \\
& q(z)' = \prod_{i} q(z_j)'
\end{split}
\end{equation}
$$

This update is performed for all latents $$j \in \{z_1,...,z_M, \theta\}$$ which gives us a new variational distribution $$q(z)'$$ that more closely approximates $$p(z)$$.

#### Maximization (M-Step)
In the M-step, we find a new value for each parameter in $$\phi$$, such that we maximize the ELBO objective with respect to the updated variational distribution $$q'$$ that we just obtained from the E step.

$$
\begin{equation}
\label{eq:m-step}
\begin{split}
\hat\phi & = \arg \max_\phi \text{ELBO}(q') \\
        & = \arg \max_\phi -\mathbb{KL}[q_j' \| \tilde{p}_{i\neq j}] + \mathbb{H}(z_{i\neq j}) + C
\end{split}
\end{equation}
$$

We keep running these two steps until the ELBO converges to a maximum.

### Conclusion
Mean-field variational inference allows us to avoid approximating the intractable evidence in a Bayesian model to obtain an approximation of the posterior distribution by optimizing a factorized variational distribution over the latents and unknown parameters of interest through expectation maximization of the evidence lower bound.

In part 2, I'll go over my implementation of mean-field VI on the task of polygenic risk score (PRS) prediction using a gaussian likelihood model with spike-and-slab prior.