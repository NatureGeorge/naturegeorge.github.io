---
layout: distill
title: Conformation 101 - Funneled energy landscape
date: 2022-05-31 15:22:00+0800
description: Thoughts on protein conformation change.
tags: conformation-101
categories: review
bibliography: 2022-05-31-conformation.bib
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU

---

When it comes to comparing modeled structures<d-footnote>experimental or predicted</d-footnote> of a protein, it is a common approach to regard structures as point clouds in $\mathbb{R}^{3}$ and quantify the difference via metrics<d-footnote>functions that follow non-degeneracy, symmetry, and triangle inequality</d-footnote> based on the difference in the Euclidean space, for instance:

* the Root Mean Square Deviation (RMSD)<d-footnote>after centered and optimal rotation: cRMSD</d-footnote>

$$
\sqrt{\frac{1}{N}\sum_{i}^{N} \lVert \mathbf{r}_{i}-\hat{\mathbf{r}}_{i} \rVert^{2}_{2}}
$$

* the distance Root Mean Square Deviation (dRMSD)<d-footnote>invariant under reflection, cannot distinguish chirality</d-footnote>

$$
\sqrt{\frac{2}{N(N-1)}\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}(d_{i,j} - \hat{d}_{i,j})^{2}}, \\
d_{i,j}=\lVert \mathbf{r}_{i}-\mathbf{r}_{j} \rVert_{2}
$$

However, molecules are connected components. When we look at a static covalent-bonded protein conformation, we focus on the backbone torsion trace and the side-chain rotamers. All the non-covalent interactions are implicit but typically explicitly evaluated through interatomic Euclidean distances. <d-footnote>In other words, this naive representation requires further augmentation of descriptions of non-covalent interactions.</d-footnote>

When movement involves, protein conformation becomes live. And the movements are characteristically described by the torsional DoF of backbone and side-chains. However, these DoF are restrained by the interatomic interactions rather than freely going around the whole space. Interactions related to solvent molecules, ligands, and binding partners should also be taken into account. Conformations without significant backbone and side-chain movements are considered to be in the same conformation cluster and trapped in the same low-energy state. On the contrary, critical movements of the backbone and side-chain are the transition process between different low-energy states.

Systematic descriptions of the energy landscapes help us to intrinsically quantify conformations and their changes. Here I will give a brief summary of pioneered works of [Prof. Jin Wang](https://www.stonybrook.edu/commcms/chemistry/faculty/_faculty-profiles/wang-jin) et al.<d-cite key="WangFoldingEvo2019"></d-cite><d-cite key="WangBindingEvo2020"></d-cite><d-cite key="WangReview2022"></d-cite> introducing the funneled energy landscape of protein structures.

If we assume the energy $E$ of a conformation (of a protein sequence) to be the sum of independent interactions, the density of states approximates a Gaussian distribution:

$$
n(E) = \frac{1}{\sqrt{2\pi\Delta E^2}}\exp\left[-\frac{(E-\bar{E})^2}{2\Delta E^2}\right],\quad\Delta E = \sqrt{\langle E^2 \rangle - \langle E \rangle^2}
$$

For a protein sequence, its total number of conformations is assumed to be $\Omega_0$. Then the conformational entropy with energy $E$ is:

$$
\begin{aligned}
S(E) &= K_B\ln(\underbrace{\Omega_{0} n(E)}_{\text{the number of conformations with energy} E}) \\
&= K_B \ln \Omega_0 + K_B \ln n(E)\\
&= \underbrace{K_B \ln \Omega_0}_{S_0} - K_B \frac{(E-\bar{E})^2}{2\Delta E^2} \color{gray}{\underbrace{- K_B \frac{\ln(2\pi\Delta E^2)}{2}}_{\text{dropped}}}
\end{aligned}
$$

Since we have the thermodynamic relation:

$$
\frac{\partial S}{\partial E} = \frac{1}{T}
$$

we can derive the most probable energy as a function of $T$ as:

$$
\begin{aligned}
\frac{-2K_{B}(E-\bar{E})}{2 \Delta E^{2}}&= \frac{1}{T}\\
&\Downarrow \\
E(T) &= \bar{E} - \frac{\Delta E^{2}}{K_{B}T} \\
\end{aligned}
$$

So the entropy at the most probable energy as a function of $T$ is:

$$
S(T) = S(E(T)) = S_{0} - \frac{\Delta E^{2}}{2K_{B}T^{2}}
$$

* (noted that) for an infinite high temperature, $S(T)$ is approximately $S_{0}$.
* (noted that) when at or below a lower enough critical temperature $T_g$ ($S(T_g)=0$), the protein is trapped in one of the frozen states.

$$
T_g = \sqrt{\frac{\Delta E^{2}}{2 K_B S_0}}
$$

From the thermodynamic expressions of the energy $E(T)$ and entropy $S(T)$, the Helmholtz free energy of the system as a function of $T$ can be expressed as:

$$
\begin{aligned}
F(T) &= E(T) - TS(T)\\
     &=\bar{E} - \frac{\Delta E^{2}}{K_{B}T} - TS_{0} + \frac{\Delta E^{2}}{2K_{B}T}\\
     &= \bar{E} - TS_{0} - \frac{\Delta E^{2}}{2K_{B}T}\\
\end{aligned}
$$

With the observations of naturally occurring proteins, we can assume that a natural protein normally has a unique ground state (with energy $E_N$) at which both the energy and entropy variance (roughness) are zero. Then the free energy of the native state equals $E_N$.

A first-order transition between native state and non-native state is expected at the temperature $T_{f}$ (folding transition temperature) where they have equal free energy. Thus we have:

$$
\begin{aligned}
  E_{N} &= \bar{E} - T_{f}S_{0} - \frac{\Delta E^{2}}{2K_{B}T_{f}} \\
  \underbrace{\bar{E}-E_{N}}_{\delta E} &= T_{f}S_{0} + \frac{\Delta E^{2}}{2K_{B}T_{f}} \\
  &\Downarrow \\
  T_{f} &= \frac{\delta E}{2S_{0}}(1+\sqrt{1 - \frac{2S_{0}\Delta E^{2}}{K_{B} \delta E^{2}}})
\end{aligned}
$$

$T_{f}$ should be larger than $T_{g}$ so as not to be trapped in the frozen state and make the system able to reach the native state. So we are focusing on $\frac{T_{f}}{T_{g}}$ and find out that:

$$
\begin{aligned}
  \frac{T_{f}}{T_{g}} &= \underbrace{\sqrt{\frac{K_{B}}{2S_{0}}}\frac{\delta E}{\Delta E}}_{\Lambda} + \sqrt{\frac{K_{B}\delta E^{2}}{2S_{0}\Delta E^{2}}-1} \\
  &= \Lambda + \sqrt{\Lambda^2 - 1} \\
\end{aligned}
$$

* The larger the ratio $\frac{T_{f}}{T_{g}}$ is, the less chance the protein has to be trapped on the way to its native state.
* $\Lambda$ itself is a quantitative measure of the landscape topogarphy. The larger the $\Lambda$ is, the more funneled protein folding energy landscape shape is against the vast number of states and roughness (i.e. minimal frustration principle).
* Maximizing $\frac{T_{f}}{T_{g}}$ is equivalent of maximizing the value of $\Lambda$.
  * This relationship connects the folding criterion to the underlying landscape topography.
  * It provides a practical implementation of the principle of minimal frustration for protein folding.
  * Optimization of $\Lambda$ guarantees the kinetic accessibility of the native state.

The probability $P(E)$ of sampling any conformation at a finite temperature $T$ with energy $E$ is:

$$
P(E) = \frac{n(E)\exp[-\frac{E-\bar{E}}{K_{B}T}]}{Z}
$$

* The probability is weighted by the Boltzmann factor $\exp[-\frac{E-\bar{E}}{K_{B}T}]$.
* $Z$ is the partition function of the canonical ensemble.

Thus, at a particular $T$ larger than $T_{g}$, the probability of the system in its unique native state and the non-native state are $P_N$ and $P_{D}$ respectively:

$$
\begin{aligned}
  P_{N}&=\frac{\exp[-\frac{E_{N}-\bar{E}}{K_{B}T}]}{Z}\\
  P_{D}&=\sum_{E>E_{N}}\frac{n(E)\exp[-\frac{E-\bar{E}}{K_{B}T}]}{Z}\\
\end{aligned}
$$

So the thermodynamic stability of the native ground state is quantified as:

$$
\begin{aligned}
  \Delta G &= -K_{B}T\ln(\frac{P_N}{P_D})\\
  &= E_N + K_{B}T\ln\left[\sum_{E>E_N} n(E) \exp(\frac{-E}{K_B T}) \right]
\end{aligned}
$$

**NOTE:**
Still on writing.

***

Cited as:

```bibtex
@online{zhu2022conformation,
        title={Conformation 101 - Funneled energy landscape},
        author={Zefeng Zhu},
        year={2022},
        month={May},
        url={https://naturegeorge.github.io/blog/2022/conformation/},
}
```
