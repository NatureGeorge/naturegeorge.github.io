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
    x! &\simeq x^{x}e^{-x}{\color{gray} \sqrt{2\pi x}} \\
    \Leftrightarrow \ln x! &\simeq x\ln x-x {\color{gray} + \frac{1}{2}\ln 2\pi x}
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
\begin{pmatrix} N \\ r \end{pmatrix} \simeq 2^{N H_{b}(r/N){\color{gray} - \frac{1}{2}\log_{2} 2\pi N (1-x) x}} \\
\text{where} \quad \underbrace{H_{b}(x) \equiv x\log_{2}\frac{1}{x} + (1-x) \log_{2}\frac{1}{1-x}}_{\text{binary entropy function}}
$$

### Euler–Maclaurin Formula

...

## Applications

### Boltzmann Factor

...