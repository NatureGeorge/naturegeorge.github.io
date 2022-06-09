---
layout: distill
title: Conformation 101 - Protein Structure Superposition
date: 2022-06-08 13:15:00+0800
description: Thoughts on protein conformation difference.
tags: conformation-101 superposition
categories: review
bibliography: 2022-06-08-struct-superposition.bib
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU

---

For measuring how much a particular atom of a molecule fluctuates over a set of conformations (e.g. simulation trajectory for a period of time or NMR ensemble), it is straightforward to calculate the Root Mean Square Fluctuation (RMSF) of this atom $\lbrace \mathbf{r}(t) \in \mathbb{R}^{3} \rbrace_{t}^{T}$ after optimal translation and rotation of the molecule:

$$
\sqrt{\frac{1}{T}\sum_{t}^{T} \lVert \mathbf{r}(t) -\langle \mathbf{r} \rangle \rVert^{2}_{2}},\quad \langle \mathbf{r} \rangle = \frac{1}{T}\sum_{t}^{T} \mathbf{r}(t)
$$

In other words, for a molecule with a set of superpositioned conformations, we can calculate the RMSF of each atom to evaluate their variations.

If we would like to measure the covariation among $K$ atoms of a molecule,
the above formula can be extended to:

$$
\sigma_{i,j} = \frac{1}{T} \sum_{t}^{T} \lVert \mathbf{r}_{i}(t) - \langle \mathbf{r}_{i} \rangle \rVert_{2} \, \lVert \mathbf{r}_{j}(t) - \langle \mathbf{r}_{j} \rangle \rVert_{2}
$$

Thus we can derive a covariance matrix $\mathbf{\Sigma}\in \mathbb{R}^{K\times K}$ describing the covariation of each of the $K$ atoms with each of the others:

$$
\mathbf{\Sigma} = \mathbf{A} \mathbf{A}^{\mathsf{T}}
$$

$$
\mathbf{A} = \begin{bmatrix}
    \lVert \mathbf{r}_{1}(1) - \langle \mathbf{r}_{1} \rangle \rVert_{2} & \dots  & \lVert \mathbf{r}_{1}(T) - \langle \mathbf{r}_{1} \rangle \rVert_{2} \\
    \lVert \mathbf{r}_{2}(1) - \langle \mathbf{r}_{2} \rangle \rVert_{2} & \dots  & \lVert \mathbf{r}_{2}(T) - \langle \mathbf{r}_{2} \rangle \rVert_{2} \\
    \vdots & \ddots & \vdots \\
    \lVert \mathbf{r}_{K}(1) - \langle \mathbf{r}_{K} \rangle \rVert_{2} & \dots  & \lVert \mathbf{r}_{K}(T) - \langle \mathbf{r}_{K} \rangle \rVert_{2}
\end{bmatrix}_{K\times T}
$$

Through normalizing the covariances by the variances, we can get the correlation matrix $\mathbf{C}$<d-footnote>Hadamard product and Hadamard power notations are used here. See [here](https://en.wikipedia.org/wiki/Hadamard_product_(matrices)) for details.</d-footnote>:

$$
\mathbf{C} = \mathbf{\Sigma} \circ (\mathbf{b}\mathbf{b}^{\mathsf{T}})^{\circ-\frac{1}{2}},\, \mathbf{b} = \mathrm{diag}(\mathbf{\Sigma})
$$

and each element $c_{i,j}$ of the correlation matrix is given by:

$$
c_{i,j} = \frac{\sigma_{i,j}}{\sqrt{\sigma_{i,i}\sigma_{j,j}}}
$$

Above mentioned calculations can be transferred from ordinary molecules to coarse-grained protein structures (e.g. define a representative atom for each kind of residue). And the conformation set of the same entity can be extended to the aligned conformation set composed of conformations from different entities (e.g. a properly superpositioned family of protein structures), in which the determination of the optimal alignment region, translation, and rotation are required in advance. Thus there are there three scenarios for the analysis of protein structures:

* quantify the internal dynamics of the same protein
* quantify the structural similarity among a protein sequence family
* quantify the structural similarity among a protein structure family

Each of the scenarios has a variant case that there may be instances with different numbers of residues or with different residue identities ([Structure Alignment Versus Structure Superposition](https://link.springer.com/chapter/10.1007/978-3-642-27225-7_8))<d-site key="TheobaldBook2012"></d-site> thus requiring appointment of alignment region. Sometimes the alignment region is (almost) deterministic, e.g. during the analysis of structure fragments of the same protein with sequence overlap or sequence family with many highly conserved residue sites. But if sequence alignment is not allowed or not feasible, we would need a structure-based alignment method, which will not be covered in this blog post but will be discussed in the near future.

Here I will first give a summary of some infrastructure works for the study of superpositioning of protein structures conducted by [Prof. Douglas L. Theobald](<https://theobald.brandeis.edu/people.php>) et al.<d-cite key="TheobaldSuperpose2006"></d-cite><d-cite key="TheobaldSuperposeSoftware2006"></d-cite><d-cite key="TheobaldSuperpose2008"></d-cite><d-cite key="TheobaldSuperpose2012"></d-cite><d-cite key="TheobaldSuperpose2019"></d-cite>

**NOTE:**
Still on writing.

***

Cited as:

```bibtex
@online{zhu2022superposition,
        title={Conformation 101 - Protein Structure Superposition},
        author={Zefeng Zhu},
        year={2022},
        month={June},
        url={https://naturegeorge.github.io/blog/2022/struct-superposition/},
}
```
