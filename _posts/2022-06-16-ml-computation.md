---
layout: post
title: Fundamental computations in ML
date: 2022-06-16 11:12:00+0800
categories: computation
description: Naive but useful keynotes on ML.
authors:
  - name: Zefeng Zhu
    affiliations:
      name: AAIS, PKU
---

## The Basic Building Blocks

### Dot Product

The dot product intrinsically defines a kind of similarity:

$$
\mathbf{a}\cdot\mathbf{b} = \sum_{i} a_{i} b_{i} = \mathbf{a}^{\mathsf{T}}\mathbf{b} = \underbrace{\lVert \mathbf{a} \rVert \lVert \mathbf{b} \rVert \cos\theta}_{\text{for orthonormal basis}}
$$

### Matrix Multiplication

And the matrix multiplication can be seen as a generalization of the dot product operation, i.e. apply dot product in batch:

$$
\begin{bmatrix}
         a_{11} & a_{12} & \cdots & a_{1n}\\
         a_{21} & a_{22} & \cdots & a_{2n}\\
         \vdots & \vdots & \ddots & \vdots\\
         a_{m1} & a_{m2} & \cdots & a_{mn}
     \end{bmatrix}
     \begin{bmatrix}
         b_{11} & b_{12} & \cdots & b_{1p}\\
         b_{21} & b_{22} & \cdots & b_{2p}\\
         \vdots & \vdots & \ddots & \vdots\\
         b_{n1} & b_{n2} & \cdots & b_{np}
     \end{bmatrix}
      =
     \begin{bmatrix}
         c_{11} & c_{12} & \cdots & c_{1p}\\
         c_{21} & c_{22} & \cdots & c_{2p}\\
         \vdots & \vdots & \ddots & \vdots\\
         c_{m1} & c_{m2} & \cdots & c_{mp}
     \end{bmatrix}
$$

$$
c_{ij}= \sum_{k=1}^n a_{ik}b_{kj}
$$

### Softmax

The Softmax serves as a smooth approximation to $$\text{onehot}(\arg\max(\mathbf{x}))$$:

$$
\text{Softmax}(\mathbf{x})=\left[\frac{\exp(x_1)}{\sum_{i}^{n}\exp(x_i)}, \ldots, \frac{\exp(x_n)}{\sum_{i}^{n}\exp(x_n)} \right]^{\mathsf{T}}
$$

## Applications

### Linear Layer

Typically, a matrix could be a representation of a linear transformation with respect to certain bases. And a linear layer (e.g. `torch.nn.Linear`) is exactly a weight matrix together with a bias vector, storing learnable parameters and representing a learnable linear transformation. Feeding an input data matrix into a linear layer, we would get a transformed data matrix.

$$
\mathbf{y} = \mathbf{Wx} + \mathbf{b}
$$

### Embedding Layer

An embedding layer (e.g. `torch.nn.Embedding`) is just a linear layer without bias but queried with onehot vectors (sparse input matrix). Thus it serves as a lookup table with learnable parameters. In deep learning frameworks, the embedding layer is optimized for retrieving with indices rather than doing matrix multiplication with sparse matrix.

### Attention (Function)

With the Query $$\mathbf{Q}\in\mathbb{R}^{n\times d_{k}}$$ and the paired Key $$\mathbf{K}\in\mathbb{R}^{m\times d_{k}}$$ and Value $$\mathbf{V}\in\mathbb{R}^{m\times d_{v}}$$, we would like to find the queries' corresponding values based on the similarity between the queries and keys. Then we can apply the Scaled Dot-Product Attention:

$$
\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Softmax}\left( \frac{\mathbf{Q}\mathbf{K}^{\mathsf{T}}}{\sqrt{d_{k}}} \right)\mathbf{V}
$$

where $$\sqrt{d_{k}}$$ is used for scaling down large dot product values.
And self-Attention i.e. $$\text{Attention}(\mathbf{X}, \mathbf{X}, \mathbf{X})$$.

### Multi-Head Attention (Layer)

*... linearly project the queries, keys and values $$h$$ times with different, learned linear projections to $$d_{k}$$, $$d_{k}$$ and $$d_v$$ dimensions, respectively.*

$$
\begin{aligned}
  \text{head}_{i} &= \text{Attention}(\mathbf{QW}_{i}^{Q}, \mathbf{KW}_{i}^{K}, \mathbf{VW}_{i}^{V}) \\
  \text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) &= \text{Concat}(\text{head}_{1}, \ldots, \text{head}_{h})\mathbf{W}^{O}
\end{aligned}
$$

Noted that $$\mathbf{W}_{i}^{Q}\in \mathbb{R}^{d_{k}\times \tilde{d_{k}}},\mathbf{W}_{i}^{K}\in \mathbb{R}^{d_{k}\times \tilde{d_{k}}},\mathbf{W}_{i}^{V}\in \mathbb{R}^{d_{v}\times \tilde{d_{v}}},\mathbf{W}^{O}\in \mathbb{R}^{h\cdot \tilde{d_{v}} \times d_{o}}$$.

*On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $$d_{v}$$-dimensional output values. These are concatenated and once again projected, resulting in the final values ... **Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.*** —— Vaswani et al. (2017). Attention Is All You Need. CoRR, abs/1706.03762.

And self Multi-Head Attention i.e. $$\text{MultiHead}(\mathbf{X}, \mathbf{X}, \mathbf{X})$$ with $$d_{v}=d_{k}, \tilde{d_{v}}=\tilde{d_{k}}$$ and embedding dimension be $$h\cdot \tilde{d_{v}}$$.

Pseudocode:

```python
x = ...
mask = ...

batch_size, seq_length, input_dim = x.size()
embed_dim = ...
num_heads = ...
head_dim = embed_dim // num_heads

qkv_proj = Linear(input_dim, 3 * embed_dim)
o_proj = Linear(embed_dim, embed_dim)

qkv = qkv_proj(x                       # from x: batch_size, seq_length, input_dim
                                       # to qkv: batch_size, seq_length, 3 * embed_dim
    ).reshape(batch_size, seq_length,  # 3 * embed_dim = 3 * head_dim * num_heads
              num_heads,               #               = num_heads * 3 * head_dim
              3 * head_dim             # batch_size, seq_length, num_heads, 3 * head_dim
    ).permute(0, 2, 1, 3)              # batch_size, num_heads, seq_length, 3 * head_dim

q, k, v = qkv.chunk(3, dim=-1)
values, attention = scaled_dot_product(q, k, v, mask=mask)
values = values.permute(0, 2, 1, 3     # batch_size, seq_length, num_heads, head_dim
                                  ).reshape(batch_size, seq_length, embed_dim) 
                                       # batch_size, seq_length, num_heads * head_dim
out = o_proj(values)
```
