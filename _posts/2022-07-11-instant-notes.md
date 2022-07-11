---
layout: distill
title: Instant Notes - ISMB 2022 3DSIG COSI
date: 2022-07-11 12:18:00+0800
description: Here are the notes on some interesting articles appeared in the ISMB 2022's 3DSIG COSI, as well as related works on the same topic.
tags: instant-notes
categories: review
bibliography: 2022-07-11-instant-notes.bib
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU
---

## Protein Comparison & Search

### Protein Structure Comparison

Applications of the Gauss Integrals on protein structure comparison considering self-intersections, proposed by Peter Røgen<d-cite key="PR2021"></d-cite><d-cite key="PR2003"></d-cite>.

### Protein Structure Search

Kempen et al. developed a new approach to perform a fast protein structure search by discretizing the tertiary interactions into structural alphabets learned by VQ-VAE<d-cite key="vanKempen2022"></d-cite>, however not mention the related works utilizing the Zernike polynomials<d-cite key="Guzenko2020"></d-cite>.

### New Sequence Alignment

Inspired by the field of protein structure contact prediction, particularly the Direct Coupling Analysis (DCA) methodology, Talibart et al. applied the Potts model considering direct couplings (i.e. coevolution) between positions in addition to positional composition (i.e. positional conservation) to align two sequences through aligning two Potts models inferred from corresponding multiple sequence alignments (MSA) via Integer Linear Programming (ILP)<d-cite key="Talibart2021"></d-cite>. Their model can be used to improve the alignment of remotely related protein sequences in tractable time. Following this idea, it is straightforward to utilize the Restricted Boltzmann Machines (RBM)<d-cite key="Monasson2019"></d-cite> and even deep neural networks to build theoretically more powerful models.

Interestingly, Petti et al. recently proposed another (similar in idea but quite different in implementation) approach to perform multiple sequence alignment<d-cite key="Petti2021"></d-cite>. They implemented a smooth and differentiable version of the Smith-Waterman pairwise alignment algorithm via differentiable dynamic programming and designed a method called Smooth Markov Unaligned Random Field (SMURF) that takes as input unaligned sequences and jointly learns an MSA. And they proved that such a differentiable alignment module helps improve the structure prediction results over those initial MSAs.

## AlphaFold2 & RoseTTAFold Downstream Analysis

### New Fold?

<d-cite key="Bordin2022"></d-cite>

### Predicting the Impact of Mutations

<d-cite key="Sen2022"></d-cite>

## Toolbox

<d-cite key="Cazals2016"></d-cite>
<d-cite key="Oteri2022"></d-cite>

***

Cited as:

```bibtex
@online{zhu2022instant-notes-on-ISMB-2022-3DSIG-COSI,
        title={Instant Notes - ISMB 2022 3DSIG COSI},
        author={Zefeng Zhu},
        year={2022},
        month={July},
        url={https://naturegeorge.github.io/blog/2022/07/instant-notes/},
}
```
