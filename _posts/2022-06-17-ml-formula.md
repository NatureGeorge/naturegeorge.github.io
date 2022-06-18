---
layout: post
title: Fundamental formulas in ML
date: 2022-06-17 14:02:00+0800
categories: computation
description: Naive but useful keynotes on ML.
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU
---

## The Basic Building Blocks

### Stirling's Approximation

$$
\begin{aligned}
    n! & \simeq n^{n} e^{-n}{\color{gray} \sqrt{2 \pi n}} \\
    \Leftrightarrow \ln n! & \simeq n\ln n-n {\color{gray} + \frac{1}{2}\ln 2\pi n}
\end{aligned}
$$

And its application on $$\begin{pmatrix} N \\ r \end{pmatrix}$$:

$$
\begin{aligned}
    \ln \begin{pmatrix}N \\ r\end{pmatrix} \equiv \ln \frac{N!}{(N-r)!r!}
&\simeq (N-r)\ln \frac{N}{N-r} + r\ln \frac{N}{r} {\color{gray} - \frac{1}{2}\ln \frac{2\pi r(N-r)}{N}} \\
&= N\left[\frac{(N-r)}{N}\ln \frac{N}{N-r} + \frac{r}{N}\ln \frac{N}{r}\right] {\color{gray} - \frac{1}{2}\ln 2\pi N \frac{(N-r)}{N}\frac{r}{N}} \\
&= N\left[\left(1-\frac{r}{N}\right)\ln \frac{1}{1-\frac{r}{N}} + \frac{r}{N}\ln \frac{N}{r}\right] {\color{gray} - \frac{1}{2}\ln 2\pi N \left(1-\frac{r}{N}\right)\frac{r}{N}} \\
&= \underbrace{N\left[\left(1-x\right)\ln \frac{1}{1-x} + x\ln \frac{1}{x}\right] {\color{gray} - \frac{1}{2}\ln 2\pi N (1-x) x}}_{x=r/N}
\end{aligned}
$$

Thus:

$$
\begin{pmatrix} N \\ r \end{pmatrix} \simeq e^{N[(1-x)\ln \frac{1}{1-x}+x\ln\frac{1}{x}]{\color{gray} - \frac{1}{2}\ln 2\pi N (1-x) x}}
$$

Noted that all the terms are logarithm and according to the Logarithm change of base rule, the above logarithm can be changed to any base. For instance:

$$
\begin{aligned}
    \begin{pmatrix} N \\ r \end{pmatrix} \simeq 2^{N H_{b}(r/N){\color{gray} - \frac{1}{2}\log_{2} 2\pi N (1-r/N) r/N}} \\
\text{where} \quad \underbrace{H_{b}(x) \equiv x\log_{2}\frac{1}{x} + (1-x) \log_{2}\frac{1}{1-x}}_{\text{binary entropy function}}
\end{aligned}
$$

### Properties of Gaussian PDF

$$
 p(\mathbf{x}|\boldsymbol{\mu,\Sigma}) = \frac{1}{\sqrt{|2\pi\boldsymbol{\Sigma}|}}\exp \Big(-\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \Big)
$$


### Euler–Maclaurin Formula

...

## Applications

### Boltzmann Factor

...

### Reparameterization

$$
\mathbb{E}_{z\sim p_{\theta}(z)}[f(z)] = \left\{ \begin{array}{rcl} \int  p_{\theta}(z) f(z) dz & \text{continuous} \\ \\ \sum_{z} p_{\theta}(z) f(z) & \text{discrete} \end{array} \right.
$$

Since the sampling process is not differentiable, we can not optimize the $p_{\theta}$ via backpropagation. We would need to convert from the expectation related to $$z$$ to the expectation related to another variable of which distribution  with no parameter to optimize.

$$
\begin{aligned}
  \mathbb{E}_{z\sim p_{\theta}(z)}[f(z)]& = \mathbb{E}_{\epsilon \sim q(\epsilon)}[f(g_{\theta}(\epsilon))] \\
  \text{where}& \quad z = g_{\theta}(\epsilon)
\end{aligned}
$$

And we have:

$$
\begin{aligned}
  \frac{\partial}{\partial \theta} \mathbb{E}_{z\sim p_{\theta}(z)}[f(z)] &= \frac{\partial}{\partial \theta} \mathbb{E}_{\epsilon \sim q(\epsilon)}[f(g_{\theta}(\epsilon))] \\
  &= \mathbb{E}_{\epsilon \sim q(\epsilon)}\left[ \frac{\partial f}{\partial g} \cdot \frac{\partial g}{ \partial \theta} \right]
\end{aligned}
$$

#### Reparameterization Trick

...

#### Gumbel-softmax Trick

...
