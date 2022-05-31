---
layout: distill
title: Conformation 101 - Different viewpoints of protein conformation change
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

When we look at a static covalent-bonded protein conformation, we focus on the backbone torsion trace and the side-chain rotamers. All the non-covalent interactions are implicit but typically explicitly evaluated through interatomic Euclidean distances.
In other words, this naive representation requires further augmentation of descriptions of non-covalent interactions.

When movement involves, protein conformation becomes live. And the movements are characteristically described by the torsional DoF of backbone and side-chains. However, these DoF are restrained by the interatomic interactions rather than freely going around the whole space. Interactions related to solvent molecules, ligands, and binding partners should also be taken into account. Conformations without significant backbone movements are considered to be in the same conformation ensemble and trapped in the same low-energy state. On the contrary, critical movements of the backbone are the transition process between different low-energy states.

Systematic descriptions of the energy landscapes help us to quantify conformations and their changes<d-cite key="PhysRevLett-122-018103"></d-cite>.

$$
n(E) = \frac{1}{\sqrt{2\pi{\Delta E}^2}}\exp\left[-\frac{(E-\bar{E})^2}{2{\Delta E}^2}\right],\quad{\Delta E} = \sqrt{\langle E^2 \rangle - \langle E \rangle^2}
$$

For a protein sequence, its total number of conformations is denoted as $\Omega_0$. We can then derive the following terms:

$$
\begin{aligned}
\Omega(E) &= \Omega_0 n(E) \\
S_0 &= K_B \log \Omega_0 \\
S(E) &= S_0 - K_B \frac{(E-\bar{E})^2}{2{\Delta E}^2}
\end{aligned}
$$

**NOTE:**
Still on writing.

Cited as:

```bibtex
@online{zhu2022conformation,
        title={Conformation 101 - Different viewpoints of protein conformation change},
        author={Zefeng Zhu},
        year={2022},
        month={May},
        url={https://naturegeorge.github.io/blog/2022/conformation/},
}
```
