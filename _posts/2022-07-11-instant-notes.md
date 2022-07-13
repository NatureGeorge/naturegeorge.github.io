---
layout: distill
title: Instant Notes - ISMB 2022 3DSIG COSI
date: 2022-07-11 12:18:00+0800
description: Here are the notes on some interesting articles appeared in the ISMB 2022's 3DSIG COSI, as well as related works on the same topic.
tags: instant-notes
categories: review
authors:
  - name: Zefeng Zhu
    url: "https://naturegeorge.github.io"
    affiliations:
      name: AAIS, PKU

bibliography: 2022-07-11-instant-notes.bib
#
toc:
  - name: Protein Comparison and Search
    #  - name: Protein Structure Comparison
    #  - name: Protein Structure Search
    #  - name: New Sequence Alignment
  - name: AlphaFold2 and RoseTTAFold Downstream Analysis
    #  - name: New Fold?
    #  - name: Predicting the Impact of Mutations
  - name: Toolbox

---

## Protein Comparison and Search

### Protein Structure Comparison

Peter Røgen presented a novel method applying the Knot theory to find topological obstructions to a superposition of one protein backbone onto another<d-cite key="PR2021"></d-cite>. Such a protein structure comparison method considers self-intersections and self-avoiding morphs. He previously utilized generalized Gauss integrals and proposed scaled Gauss metric as geometric measures of protein structures<d-cite key="PR2003"></d-cite>.

### Protein Structure Search

Kempen et al. developed a new approach to perform a fast protein structure search by discretizing the tertiary interactions into structural alphabets learned by VQ-VAE<d-cite key="vanKempen2022"></d-cite> and emphasized advantages over those discretizing the local backbone<d-cite key="Brevern2000"></d-cite> (related to Alexandre G. de Brevern group's works),

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://www.biorxiv.org/content/biorxiv/early/2022/06/24/2022.02.07.479398/F1.large.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig 1 of Kempen et al.
</div>

however not mention the related works utilizing the 3D Zernike polynomials that supporting both monomeric and oligomeric query<d-cite key="Guzenko2020"></d-cite>.

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://journals.plos.org/ploscompbiol/article/figure/image?size=medium&id=10.1371/journal.pcbi.1007970.g003" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig 3 of Guzenko et al.
</div>

For discretizing tertiary interactions, there are also some related works<d-cite key="Shi2014"></d-cite><d-cite key="Jure2022"></d-cite><d-cite key="Nepomnyachiy2017"></d-cite>,

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.0083788.g001" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.0083788.g003" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://journals.plos.org/plosone/article/figure/image?size=medium&id=10.1371/journal.pone.0263566.g001" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Fig 1 and Fig 3 of Shi et al. & Fig 1 of Pražnikar et al.
</div>

particularly Gevorg Grigoryan group's works<d-cite key="Zheng2015"></d-cite><d-cite key="Mackenzie2016"></d-cite><d-cite key="Zhou2020"></d-cite>.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://els-jbs-prod-cdn.jbs.elsevierhealth.com/cms/attachment/943141aa-8fb6-44e2-a51a-fa35e51afd24/fx1.jpg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://www.pnas.org/cms/10.1073/pnas.1607178113/asset/e0d64f5b-0d4b-4882-9d9b-dc39866a0048/assets/graphic/pnas.1607178113fig01.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="https://www.pnas.org/cms/10.1073/pnas.1908723117/asset/84ce450d-1d76-4fcc-9717-03aac5f502d0/assets/graphic/pnas.1908723117fig01.jpeg" class="img-fluid rounded z-depth-1" zoomable=true %}
    </div>
</div>
<div class="caption">
    Summary Fig of Zheng et al. & Fig 1 of Mackenzie et al. & Fig 1 of Zhou et al.
</div>

### New Sequence Alignment

Inspired by the field of protein structure contact prediction, particularly the Direct Coupling Analysis (DCA) methodology, Talibart et al. applied the Potts model considering direct couplings (i.e. coevolution) between positions in addition to positional composition (i.e. positional conservation) to align two sequences through aligning two Potts models inferred from corresponding multiple sequence alignments (MSA) via Integer Linear Programming (ILP)<d-cite key="Talibart2021"></d-cite>. Their model can be used to improve the alignment of remotely related protein sequences in tractable time. Following this idea, it is straightforward to utilize the Restricted Boltzmann Machines (RBM)<d-cite key="Monasson2019"></d-cite> and even deep neural networks to build theoretically more powerful models.

Interestingly, Petti et al. recently proposed another (similar in idea but quite different in implementation) approach to perform multiple sequence alignment<d-cite key="Petti2021"></d-cite>. They implemented a smooth and differentiable version of the Smith-Waterman pairwise alignment algorithm via differentiable dynamic programming and designed a method called Smooth Markov Unaligned Random Field (SMURF) that takes as input unaligned sequences and jointly learns the MSA. And they proved that such a differentiable alignment module helps improve the structure prediction results over those initial MSAs.

## AlphaFold2 and RoseTTAFold Downstream Analysis

### New Fold?

Bordin et al. reported a new CATH-Assign protocol (ultizing Foldseek<d-cite key="vanKempen2022"></d-cite> for fast structure comparison) which is used to analyze the AlphaFoldDB and detect new superfamilies<d-cite key="Bordin2022"></d-cite>. It seems that AlphaFold2 yields a certain amount of "novel" structures. But people should be cautious about this since the predicted structures are not always "true" and the structure comparison methods may not be robust enough<d-cite key="PR2021"></d-cite>.

### Predicting the Impact of Mutations

Sen et al. used both AlphaFold and RoseTTAFold to predict the structures of protein domains without known experimental structures, and perform subsequent functional predictions based on those predicted structures to help estimate the effect of disease-associated missense mutations<d-cite key="Sen2022"></d-cite>. Such incorporating two models to try to yield better results is a kind of ensemble approach.

## Toolbox

* The structural bioinformatics library (SBL)<d-cite key="Cazals2016"></d-cite>
* iBIS2Analyzer: a web server for a phylogeny-driven coevolution analysis of protein families<d-cite key="Oteri2022"></d-cite>

Require further investigation for usability.

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
