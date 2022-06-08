---
layout: distill
title: Conformation 101 - Protein Structure Evolution
date: 2022-06-08 13:15:00+0800
description: Thoughts on protein conformation change.
tags: conformation-101 evolution
categories: review
bibliography: 2022-06-08-struct-evolution.bib
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU

---

Here I will give a brief summary of some infrastructure works for the study of the evolution of protein structure conducted by [Prof. Douglas L. Theobald](<https://theobald.brandeis.edu/people.php>) et al.<d-cite key="TheobaldSuperpose2008"></d-cite><d-cite key="TheobaldSuperpose2012"></d-cite><d-cite key="TheobaldSuperpose2019"></d-cite><d-cite key="TheobaldStructEvo2021"></d-cite>

For measuring how much a particular atom of a molecule fluctuates over a set of conformations (e.g. simulation trajectory for a period of time or NMR ensemble), it is straightforward to calculate the Root Mean Square Fluctuation (RMSF) of this atom $\{\mathbf{r}(t) \in \mathbb{R}^{3} \}_{t}^{T}$ after optimal translation and rotation of the molecule:

$$
\sqrt{\frac{1}{T}\sum_{t}^{T} \lVert \mathbf{r}(t) -\langle \mathbf{r} \rangle \rVert^{2}_{2}},\quad \langle \mathbf{r} \rangle = \frac{1}{T}\sum_{t}^{T} \mathbf{r}(t)
$$

In other words, for a molecule with a set of superpositioned conformations, we can calculate the RMSF of each atom to evaluate their variations.

If we would like to measure the covariation among $N$ atoms of a molecule,
the above formula can be extended to:

$$
\sigma_{i,j} = \frac{1}{T} \sum_{t}^{T} \lVert \mathbf{r}_{i}(t) - \langle \mathbf{r}_{i} \rangle \rVert_{2} \, \lVert \mathbf{r}_{j}(t) - \langle \mathbf{r}_{j} \rangle \rVert_{2}
$$

Thus we can derive a covariance matrix $\Sigma\in \mathbb{R}^{N\times N}$ describing the covariation of each of the $N$ atoms with each of the others.

Through normalizing the covariances by the variances, we can get the correlation matrix $\mathbf{C}$, and each element $c_{i,j}$ of the correlation matrix is given by:

$$
c_{i,j} = \frac{\sigma_{i,j}}{\sqrt{\sigma_{i,i}\sigma_{j,j}}}
$$

Above mentioned calculations can be transferred from ordinary molecules to coarse-grained protein structures (e.g. define a representative atom for each kind of residue). And the conformation set of the same entity can be extended to the aligned conformation set composed of conformations from different entities (e.g. a properly superpositioned family of protein structures), in which the determination of the optimal alignment region, optimal translation, and optimal rotation are required in advance.

**NOTE:**
Still on writing.

***

Cited as:

```bibtex
@online{zhu2022evolution,
        title={Conformation 101 - Protein Structure Evolution},
        author={Zefeng Zhu},
        year={2022},
        month={June},
        url={https://naturegeorge.github.io/blog/2022/struct-evolution/},
}
```
