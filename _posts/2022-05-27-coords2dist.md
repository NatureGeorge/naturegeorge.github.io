---
layout: post
title: Ways to compute distance map
date: 2022-05-27 15:50:00+0800
categories: coding
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU
---

Just take advantage of the symmetric property:

{% highlight python linenos %}

import torch
coords = ...  # (N, 3)

idx = torch.triu_indices(coords.shape[0], coords.shape[0], offset=1)
D = torch.zeros(coords.shape[0], coords.shape[0], dtype=torch.float)
D[idx[0], idx[1]] = D[idx[1], idx[0]] = (coords[idx[0]] - coords[idx[1]]).norm(dim=1)

{% endhighlight %}

Or do it more algebraically:

$$
\begin{aligned}
\mathbf{B} &= \mathbf{X}\mathbf{X}^{\mathsf{T}} \\
\mathbf{c} &= \mathrm{diag}(\mathbf{B}) \\
\mathbf{D} &= (\mathbf{c}\mathbf{1}^{\mathsf{T}} + \mathbf{1}\mathbf{c}^{\mathsf{T}} - 2\mathbf{B})^{\circ \frac{1}{2}}
\end{aligned}
$$

{% highlight python linenos %}

B = coords @ coords.T
c = torch.diag(B).expand(coords.shape[0], coords.shape[0])
D = torch.sqrt(-2 * B + c + c.T)

{% endhighlight %}

If radius cutoff (e.g. 8) is known in prior:

{% highlight python linenos %}

from torch_geometric.nn import radius_graph

idx = radius_graph(coords, r=8.0)

{% endhighlight %}
