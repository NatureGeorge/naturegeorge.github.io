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

### Taylor Series Approximation

Around a given point $$x_{0}$$, we have:

$$
\begin{aligned}
  f(x) &= \sum_{n=0}^{\infty}\frac{f^{(n)}(x_{0})}{n!}(x-x_{0})^{n}\\
       &\simeq \underbrace{\left(\sum_{n=0}^{k}\frac{f^{(n)}(x_{0})}{n!}(x-x_{0})^{n}\right)}_{k\text{th order Taylor polynomial}} + \underbrace{\frac{f^{(k+1)}(c)}{(k+1)!}(x-x_{0})^{k+1}}_{\text{remainder (mean-value (Lagrange) form)}}
\end{aligned}
$$

which requires $$f$$ to be a k+1 times differentiable function.

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

### Soft Maximum

One of the approaches is:

$$
\begin{aligned}
  \max(x,y) &\simeq \ln(\exp(x)+\exp(y))\\
  \mathrm{abs}(x) = \max(x,-x) &\simeq \ln(\exp(x)+\exp(-x))\\
  \max(\mathbf{x}) &\simeq \ln\left(\sum_{i=1}^{n} \exp(x_{i}) \right) \triangleq \mathrm{logsumexp}(\mathbf{x})\\
\end{aligned}
$$

For values with a larger gap, this approximation would be more precise, as the gap between those exponential values would be larger making the logarithm of the sum closer to the maximum value. So we can add a coefficient to adjust the scale of input values depending on the precision we would like to achieve:

$$
\max(\mathbf{x}) \simeq \frac{1}{k}\ln\left(\sum_{i=1}^{n} \exp(k x_{i}) \right)
$$

But it should be noted that this approximation is easy to overflow or underflow for computers. We can shift the input values by a constant:

$$
\ln( \exp(x) + \exp(y)) = \ln( \exp(x – c) + \exp(y–c) ) + c
$$

And we get:

$$
\begin{aligned}
  \max(\mathbf{x}) &\simeq \frac{1}{k} \left[\ln\left(\sum_{i=1}^{n} \exp(k x_{i} - c) \right) + c \right] \\
  \text{where } & c=\max(k\mathbf{x})
\end{aligned}
$$

### Matrix Exponential

$$
\exp (\mathbf{A}) = \sum^{\infty}_{n=0} \frac{\mathbf{A}^{n}}{n!}
$$

$$
\begin{aligned}
  \det(\exp (\mathbf{A})) &= \exp(\mathrm{Tr}(\mathbf{A})) \\
  \ln(\det(\underbrace{\mathbf{B}}_{\exp(\mathbf{A})})) &= \mathrm{Tr}(\ln(\mathbf{B}))
\end{aligned}
$$

### Properties of Gaussian Distribution

$$
 p(\mathbf{x};\boldsymbol{\mu,\Sigma}) = \frac{1}{\sqrt{|2\pi\boldsymbol{\Sigma}|}}\exp \Big(-\frac{1}{2} (\mathbf{x}-\boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu}) \Big)
$$

## Applications

### Numerical Integration and Differentiation

...

### Boltzmann Factor

...

### Reparameterization

$$
\begin{aligned}
\mathbb{E}_{z\sim p_{\theta}(z)}[f(z)] &= \left\{ \begin{array}{rcl} \int  p_{\theta}(z) f(z) dz & \text{continuous} \\ \\ \sum_{z} p_{\theta}(z) f(z) & \text{discrete} \end{array} \right. \\
&\approx \frac{1}{n} \sum_{z} f(z)
\end{aligned}
$$

Since the sampling process is not differentiable, we can not optimize the $$p_{\theta}$$ via methods like backpropagation. We would need to convert from the expectation related to $$z$$ to the expectation related to another variable of which distribution  with no parameter to optimize.

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
  &= \mathbb{E}_{\epsilon \sim q(\epsilon)}\left[ \frac{\partial f}{\partial g} \cdot \frac{\partial g_{\theta}(\epsilon)}{ \partial \theta} \right]
\end{aligned}
$$

### Kernel Method

#### Kernel Function

...

#### Kernel Trick

...

### Representation of Transformations

$$
\begin{bmatrix}
  \cos n\theta & -\sin n \theta \\
  \sin n\theta & \cos n\theta
\end{bmatrix} = \exp \left( n\theta \begin{bmatrix}
  0 & -1 \\ 1 & 0
\end{bmatrix} \right)
$$
