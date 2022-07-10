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

> Notation: bold, lower-case letters refer to column vectors

## The Basic Building Blocks

### Norm

...

### Dot Product

The dot product intrinsically defines a kind of similarity:

$$
\mathbf{a}\cdot\mathbf{b} = \sum_{i} a_{i} b_{i} = \mathbf{a}^{\mathsf{T}}\mathbf{b} = \underbrace{\lVert \mathbf{a} \rVert_{2} \lVert \mathbf{b} \rVert_{2} \cos\theta}_{\text{for orthonormal basis}}
$$

And it is the way to perform vector projection:

$$
\mathrm{proj}_{\mathbf{a}}\mathbf{b} = \frac{(\mathbf{a}^{\mathsf{T}}\mathbf{b}) \mathbf{a}}{\mathbf{a}^{\mathsf{T}}\mathbf{a}}=\frac{\mathbf{a}\mathbf{a}^{\mathsf{T}}\mathbf{b}}{\mathbf{a}^{\mathsf{T}}\mathbf{a}}
$$

and make it simple to derive the trigonometric formulas:

$$
\begin{array}{c}
  \mathbf{a} = \begin{bmatrix} \cos\alpha \\ \sin\alpha \end{bmatrix}, \mathbf{b} = \begin{bmatrix} \cos\beta \\ \sin\beta \end{bmatrix} \quad\text{(unit circle)} \\
  \mathbf{a}\cdot\mathbf{b} = \cos\alpha\cos\beta + \sin\alpha\sin\beta = \cos(\alpha - \beta) = \cos(\beta-\alpha)
\end{array}
$$

Noted that we have the Cauchy-Buniakowsky-Schwarz Inequality:

$$
\left(\sum_{i} a_{i} b_{i}\right)^{2} \le \left(\sum_{i} a_{i}^{2}\right) \left(\sum_{i} b_{i}^{2}\right)
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

Then we can derive the outer product operation of two column vectors:

$$
\mathbf{u} \otimes \mathbf{v} = \mathbf{uv}^{\mathsf{T}}= \begin{bmatrix}
         u_{1}v_{1} & u_{1}v_{2} & \cdots & u_{1}v_{n}\\
         u_{2}v_{1} & u_{2}v_{2} & \cdots & u_{2}v_{n}\\
         \vdots & \vdots & \ddots & \vdots\\
         u_{m}v_{1} & u_{m}v_{2} & \cdots & u_{m}v_{n}
     \end{bmatrix}
$$

The matrix multiplication can also be formulated as:

$$
\mathbf{A}_{m\times n}\mathbf{B}_{n\times p}=\sum_{k=1}^{n} \mathbf{A}_{(:,k)}\mathbf{B}_{(k,:)}
$$

We can find out that there exist ways to perform block matrix multiplication:

$$
\mathbf{A} = \begin{bmatrix}
    \mathbf{M}_{ra} & \mathbf{M}_{rb} \\
    \mathbf{M}_{sa} & \mathbf{M}_{sb}
  \end{bmatrix},
\mathbf{B} = \begin{bmatrix}
    \mathbf{M}_{at} & \mathbf{M}_{au} \\
    \mathbf{M}_{bt} & \mathbf{M}_{bu}
  \end{bmatrix}
$$

$$
\mathbf{AB} = \begin{bmatrix}
    \mathbf{M}_{ra}\mathbf{M}_{at} + \mathbf{M}_{rb}\mathbf{M}_{bt}
& \mathbf{M}_{ra}\mathbf{M}_{au} + \mathbf{M}_{rb}\mathbf{M}_{bu} \\
    \mathbf{M}_{sa}\mathbf{M}_{at} + \mathbf{M}_{sb}\mathbf{M}_{bt}
& \mathbf{M}_{sa}\mathbf{M}_{au} + \mathbf{M}_{sb}\mathbf{M}_{bu}
  \end{bmatrix}
= \begin{bmatrix}
    \mathbf{M}_{rt} & \mathbf{M}_{ru} \\
    \mathbf{M}_{st} & \mathbf{M}_{su}
  \end{bmatrix}
$$

### Matrix Decomposition

#### LU Decomposition

...

#### QR Decomposition

For $$ \mathbf{A} \in \mathbb{R}^{m\times n}\quad(m\ge n) $$, through the Gram-Schmidt process (MGS) or Givens rotation or Householder transformation, we have:

$$
\begin{aligned}
  \mathbf{A}_{m\times n} &=\underbrace{\mathbf{Q}_{m\times n}}_{\text{n orthonormal column vectors}}\underbrace{\mathbf{R}_{n\times n}}_{\text{triu}} \\
  \text{where } & \mathbf{Q}^{\mathsf{T}}\mathbf{Q} \in \mathbf{I}_{n}
\end{aligned}
$$

Noted that:

$$
\begin{aligned}
   \mathbf{A}^{\mathsf{T}}\mathbf{A} &= (\mathbf{QR})^{\mathsf{T}}(\mathbf{QR}) \\
   &= \mathbf{R}^{\mathsf{T}}\mathbf{Q}^{\mathsf{T}}\mathbf{QR} \\
   &= \mathbf{R}^{\mathsf{T}} \mathbf{R} \\
   \Rightarrow \det \mathbf{A}^{\mathsf{T}}\mathbf{A} &= \det \mathbf{R}^{\mathsf{T}} \mathbf{R} \\
   &=\det \mathbf{R}^{\mathsf{T}} \det \mathbf{R} \\
   &= (\det \mathbf{R})^2
\end{aligned}
$$

#### Eigenvalue Decomposition

$$
\det(\mathbf{A}-\lambda \mathbf{I})
$$

For $$\mathbf{A}_{1} \sim \mathbf{A}_{2}$$ (i.e. $$\mathbf{A}_{1}=\mathbf{SA}_{2}\mathbf{S}^{-1}$$):

$$
\begin{aligned}
  \mathbf{A}_{1}-\lambda \mathbf{I} &= \mathbf{SA}_{2}\mathbf{S}^{-1}-\lambda \mathbf{I} = \mathbf{S}(\mathbf{A}_{2}-\lambda \mathbf{I})\mathbf{S}^{-1} \\
  \det(\mathbf{A}_{1}-\lambda \mathbf{I}) &= \det(\mathbf{S})\det(\mathbf{A}_{2}-\lambda \mathbf{I})\det(\mathbf{S}^{-1}) =\det(\mathbf{A}_{2}-\lambda \mathbf{I})
\end{aligned}
$$

#### Singular Value Decomposition

...

(Moore–Penrose inverse)

### Matrix Derivatives

$$
\begin{aligned}
  \frac{\partial \lVert \mathbf{a} - \mathbf{b} \rVert_{2}}{\partial \mathbf{a}} &= \frac{\mathbf{a}-\mathbf{b}}{\lVert \mathbf{a} - \mathbf{b} \rVert_{2}}
\\
\frac{\partial \lVert \mathbf{a} - \mathbf{b} \rVert_{2}^{2}}{\partial \mathbf{a}} &= 2(\mathbf{a}-\mathbf{b})
\end{aligned}
$$

### Matrix Lie Groups

#### Rotations

For basis vectors

$$
\mathbf{e}_{1}=\begin{bmatrix} 1\\0 \end{bmatrix},\mathbf{e}_{2}=\begin{bmatrix} 0\\1 \end{bmatrix}
$$

, a rotation of the plane $$\mathbb{R}^{2}$$ about the origin $$O$$ through angle $$\theta$$ is a linear
transformation $$R_{\theta}: \mathbb{R}^{2} \rightarrow \mathbb{R}^{2}$$ that making

$$
\begin{aligned}
  R_{\theta}(\mathbf{e}_{1}) &= \begin{bmatrix}\cos\theta \\ \sin\theta \end{bmatrix} \\
R_{\theta}(\mathbf{e}_{2}) &= \begin{bmatrix}\cos(\theta+\frac{\pi}{2}) \\ \sin(\theta+\frac{\pi}{2}) \end{bmatrix} = \begin{bmatrix} -\sin\theta \\ \cos\theta \end{bmatrix}
\end{aligned}
$$

Thus:

$$
\begin{aligned}
  \mathbf{R}_{\theta} &= \begin{bmatrix}
  \cos\theta & -\sin\theta\\
  \sin\theta & \cos\theta
\end{bmatrix} \\ &= \cos\theta \underbrace{\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}}_{\mathbf{1}} + \sin\theta \underbrace{\begin{bmatrix} 0 & -1 \\ 1 & 0 \end{bmatrix}}_{\mathbf
i} \\
  \mathbf{R}_{\theta}\begin{bmatrix} x\\ y\end{bmatrix} &=
  \begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{bmatrix} \\
  \mathbf{R}_{\theta}\mathbf{R}_{\beta} &= \mathbf{R}_{\theta+\beta} = \mathbf{R}_{\beta}\mathbf{R}_{\theta}
\end{aligned}
$$

If we take the complex view, each rotation $$R_{\theta}$$ of $$\mathbb{R}^{2}$$ can be represented by the complex number:

$$
\begin{aligned}
  z_{\theta} &=\underbrace{\cos\theta+i\sin\theta}_{e^{i\theta}} \\
  z_{\theta}(x+iy) &= (\cos\theta+i\sin\theta)(x+iy) \\
  &= (x\cos\theta - y\sin\theta) + i (x\sin\theta + y\cos\theta)\\
  z_{\theta}z_{\beta} &= e^{i(\theta+\beta)} = z_{\theta+\beta} = z_{\beta}z_{\theta}
\end{aligned}
$$

These two representations of rotations have intrinsic relationships. Noted that all complex numbers have corresponding 2×2 real matrices:

$$
a+bi \Leftrightarrow \begin{bmatrix} a & -b \\ b & a \end{bmatrix} = a \mathbf{1} + b\mathbf{i},\quad \text{where }a,b\in \mathbb{R}
$$

We can view it as this: we model the real unit $$1$$ by $$(\mathbf{e}_{1})^{\star}=\mathbf{1}$$ and model the imaginary unit $$i$$ by $$(\mathbf{e}_{2})^{\star}=\mathbf{i}$$, where

$$
\begin{aligned}
  \begin{bmatrix}a\\b\end{bmatrix}^{\star} &= \begin{bmatrix} a & -b \\ b & a \end{bmatrix}\\
  c\left(\begin{bmatrix}a\\b\end{bmatrix}^{\star}\right)&=\left(c\begin{bmatrix}a\\b\end{bmatrix}\right)^{\star}
\end{aligned}
$$

So, for $$a+bi$$ the real part is modeled by $$a(\mathbf{e}_{1})^{\star}=(a\mathbf{e}_{1})^{\star}$$ and the imaginary part is modeled by $$b(\mathbf{e}_{2})^{\star}=(b\mathbf{e}_{2})^{\star}$$. Thus the complex number is modeled by:

$$
\begin{aligned}
  a(\mathbf{e}_{1})^{\star}+b(\mathbf{e}_{2})^{\star} &= (a\mathbf{e}_{1})^{\star}+(b\mathbf{e}_{2})^{\star} \\
  &= (a\mathbf{e}_{1}+b\mathbf{e}_{2})^{\star}
\end{aligned}
$$

The matrix representation behave exactly the same as the complex numbers under addition and multiplication (matrices as $$\mathbb{C}$$):

$$
\begin{aligned}
  (a \mathbf{1} + b\mathbf{i})(c \mathbf{1} + d\mathbf{i})&=\begin{bmatrix} a & -b \\ b & a \end{bmatrix} \begin{bmatrix} c & -d \\ d & c \end{bmatrix} \\&
  = \begin{bmatrix} ac-bd & -(ad+bc) \\ ad+bc & ac-bd \end{bmatrix} \\
  &=(ac-bd) \mathbf{1} + (ad+bc)\mathbf{i}
\end{aligned}
$$

$$
\begin{aligned}
  \mathbf{1}^{2}=\mathbf{1},\mathbf{1i}&=\mathbf{i1}=\mathbf{i},\mathbf{i}^{2}=\mathbf{-1} \\
  (a \mathbf{1} + b\mathbf{i})^{-1} &= \frac{a \mathbf{1} - b\mathbf{i}}{a^{2}+b^{2}}\\
\det(a\mathbf{1}+b\mathbf{i}) &= a^{2}+b^{2} = \lvert a+bi \rvert^{2} \\
\det((a \mathbf{1} + b\mathbf{i})(c \mathbf{1} + d\mathbf{i})) &=\det(a \mathbf{1} + b\mathbf{i})\det(c \mathbf{1} + d\mathbf{i}) \\
=(ac-bd)^{2}+(ad+bc)^{2} &= (a^{2}+b^{2})(c^{2}+d^{2})\\
= \lvert (a+bi)(c+di) \rvert^{2} &= \lvert a+bi \rvert^{2} \lvert c+di \rvert^{2}
\end{aligned}
$$

So, for rotations, we have:

$$
\begin{aligned}
  z_{\theta}(x+iy) \Leftrightarrow &\quad (\underbrace{\cos\theta\mathbf{1} +\sin\theta \mathbf{i}}_{\mathbf{R}_{\theta}})(x\mathbf{1}+y\mathbf{i}) \\
  &= (x\cos\theta - y\sin\theta)\mathbf{1} + (x\sin\theta + y\cos\theta)\mathbf{i}\\
  &=\begin{bmatrix} x\cos\theta - y\sin\theta \\ x\sin\theta + y\cos\theta \end{bmatrix}^{\star}
\end{aligned}
$$

If we consider the set of all rotations of the plane $$\mathbb{R}^{2}$$ about the origin $$O$$, the set is the special orthogonal group for n=2:

$$
\mathrm{SO}(2) = \lbrace \mathbf{R} \in \mathbb{R}^{2 \times 2} | \mathbf{R}\mathbf{R}^{\mathsf{T}}=\mathbf{I},\det(\mathbf{R})=1 \rbrace
$$

If we consider the set of all rotations of the plane $$\mathbb{C}$$ about the origin $$O$$, the set is the unit circle or 1-dimensional sphere (also a (Lie) group under the operation of complex number multiplication):

$$
\mathbb{S}^{1} = \lbrace z: \lvert z \rvert = 1 \rbrace
$$

The complex numbers are ordered pairs with the *sum,* *product,* and *absolute value* operations that can express through matrix computations. We can extend this idea to the ordered quadruples of real values with the matrix:

$$
\begin{array}{c}
  q = \begin{bmatrix}
  a+id & -b-ic \\
  b-ic & a-id
\end{bmatrix}
= a\mathbf{1}+b\mathbf{i}+c\mathbf{j}+d\mathbf{k}\\
\text{where } \mathbf{j}=\begin{bmatrix}0 & -i \\ -i & 0\end{bmatrix}, \mathbf{k}=\begin{bmatrix} i & 0 \\ 0 & -1\end{bmatrix}
\end{array}
$$

This is one of the forms of the quaternions.

### Optimization

...

### Softmax

The Softmax serves as a smooth approximation to $$\text{onehot}(\arg\max(\mathbf{x}))$$:

$$
\text{Softmax}(\mathbf{x})=\left[\frac{\exp(x_1)}{\sum_{i}^{n}\exp(x_i)}, \ldots, \frac{\exp(x_n)}{\sum_{i}^{n}\exp(x_n)} \right]^{\mathsf{T}}
$$

## Applications

### Standardizing and Whitening Data

For a data matrix $$\mathbf{X}\in\mathbb{R}^{N\times D}$$ (row: sample, column: feature), we would like to:

* standardize the data: *to ensure the features are comparable in magnitude, which can help with model fitting and inference*
* whiten the data: to remove correlation between the features

The most common way to standardize the data is:

$$
\begin{aligned}
  \text{standardize}(x_{nd}) &= \frac{x_{nd}-\hat{\mu}_{d}}{\hat{\sigma}_{d}} \\
  &\simeq \frac{x_{nd}-\hat{\mu}_{d}}{\sqrt{\hat{\sigma}_{d}^{2}+\epsilon}}
\end{aligned}
$$

For each of the features, this Z-score standardization centers the mean to zero and scales the data to unit variance (i.e. remove bias and scale).

However, standardization does not consider the covariation between features. Thus we have to whiten the data matrix via linear transformation (on feature dimension) so as to make the covariance matrix become an identity matrix (as long as the covariance matrix is nonsingular). Suppose the data matrix is already centered, we have:

$$
\begin{aligned}
  \mathbf{\Sigma} &= \frac{1}{N}\mathbf{X}^{\mathsf{T}}\mathbf{X} \\
  \mathbf{I} &= \frac{1}{N} \mathbf{W}^{\mathsf{T}}\mathbf{X}^{\mathsf{T}}\mathbf{XW} = \mathbf{W}^{\mathsf{T}}\mathbf{\Sigma}\mathbf{W}
\end{aligned}
$$

For preserving the original characteristics of the data matrix, we can define $$\mathbf{W} \in \mathbb{R}^{D\times D}, \det\mathbf{W}>0$$ and get:

$$
\mathbf{\Sigma} = (\mathbf{W}^{\mathsf{T}})^{-1}\mathbf{W}^{-1} =  (\mathbf{W}^{-1})^{\mathsf{T}}\mathbf{W}^{-1}
$$

Since the covariance matrix is a real symmetric matrix, it is always orthogonally diagonalizable:

$$
\mathbf{\Sigma} = \mathbf{V\Lambda V}^{\mathsf{T}}
$$

Thus we can let:

$$
\mathbf{W}^{-1} = \underbrace{\mathbf{R}}_{\text{arbitrary orthogonal matrix}}\sqrt{\mathbf{\Lambda}}\mathbf{V}^{\mathsf{T}}
$$

and get:

$$
\begin{aligned}
  \mathbf{W} &= \mathbf{V}\sqrt{\mathbf{\Lambda}^{-1}}\mathbf{R}^{\mathsf{T}} \\
  &= \left\{ \begin{array}{lll} \mathbf{V}\sqrt{\mathbf{\Lambda}^{-1}},&\mathbf{R} = \mathbf{I} &\text{(PCA whitening)} \\ \mathbf{V}\sqrt{\mathbf{\Lambda}^{-1}}\mathbf{V}^{\mathsf{T}},&\mathbf{R} = \mathbf{V} &\text{(ZCA whitening)} \end{array} \right.
\end{aligned}
$$

Noted that it would require a regularization term $$\epsilon$$ added to the eigenvalues so as to avoid dividing by zero.

### Linear Layer

Typically, a matrix could be a representation of a linear transformation with respect to certain bases. And a linear layer (e.g. `torch.nn.Linear`) is exactly a weight matrix together with a bias vector, storing learnable parameters and representing a learnable linear transformation. Feeding an input data matrix into a linear layer, we would get a transformed data matrix.

$$
\begin{array}{llll}
  \mathbf{y} = \mathbf{Wx} + \mathbf{b}, &\mathbf{x} \in \mathbb{R}^{d \times 1},&\mathbf{W} \in \mathbb{R}^{\hat{d} \times d}, &\mathbf{b},\mathbf{y} \in \mathbb{R}^{\hat{d}\times 1} \\
  \mathbf{Y} = \mathbf{XW} + \mathbf{B}, &\mathbf{X} \in \mathbb{R}^{n \times d}, &\mathbf{W}\in \mathbb{R}^{d\times\hat{d}}, &\mathbf{B},\mathbf{Y}\in \mathbb{R}^{n\times \hat{d}}
\end{array}
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

*On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $$d_{v}$$-dimensional output values. These are concatenated and once again projected, resulting in the final values ... **Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.*** —— Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A., Kaiser, u., & Polosukhin, I. (2017). Attention is All you Need. In Advances in Neural Information Processing Systems. Curran Associates, Inc.

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

### Least Square

$$
\mathbf{Ax}=\mathbf{b}
$$

$$
\lVert \mathbf{Ax}-\mathbf{b} \rVert_{2}
$$

### Nonlinear Least Square

...

### Group Equivariance

* Let discriminator function denoted as $$f:\mathbb{R}^{d} \rightarrow \mathbb{R}$$, group operator denoted as $$g \in G$$
  * then group invariance can be expressed as: $$f(\mathbf{x})=f(g(\mathbf{x}))$$
  * we say that such function is invariant to $$g$$ since with and without the group action results in the same output
* Let discriminator function denoted as $$f:\mathbb{R}^{d} \rightarrow \mathbb{R}^{d'}$$, group operator in input space denoted as $$g \in G$$, group operator in output space denoted as $$g' \in G'$$
  * then group equivariance can be expressed as: $$f(g(\mathbf{x}))=g'(f(\mathbf{x}))$$
  * we say that such function is equivariant to $$g$$ since there exists an equivalent transformation $$g'$$
on its output space

$$
\begin{array}{lll}
  &\mathbf{x} &\xrightarrow[f]{} & f(\mathbf{x}) \\
  &\big\downarrow^{g\in G} & &\big\downarrow^{g'\in G'} \\
  &g(\mathbf{x}) &\xrightarrow[f]{} & \left\{ \begin{array}{r} g'(f(\mathbf{x})) \\ f(g(\mathbf{x})) \end{array} \right.
\end{array}
$$

### Lie Algebra Convolutional Layer

*...* —— Dehmamy, N., Walters, R., Liu, Y., Wang, D., & Yu, R. (2021). Automatic Symmetry Discovery with Lie Algebra Convolutional Network. In Advances in Neural Information Processing Systems (pp. 2503–2515). Curran Associates, Inc.

### E(n)-Equivariant Graph Convolutional Layer

*...* —— Satorras, V., Hoogeboom, E., & Welling, M. (2021). E(n) Equivariant Graph Neural Networks. In Proceedings of the 38th International Conference on Machine Learning (pp. 9323–9332). PMLR.

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

#### Reparameterizing Distributions on Lie Groups

*...* —— Falorsi, L., de Haan, P., Davidson, T., & Forré, P. (2019). Reparameterizing Distributions on Lie Groups. In Proceedings of the Twenty-Second International Conference on Artificial Intelligence and Statistics (pp. 3244–3253). PMLR.
