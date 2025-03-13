+++
date = '2025-02-28T19:49:18+05:30'
draft = false
title = 'Linear Regression From Scratch'
math = true
description = 'In depth explanation of one of the most fundamental Machine Learning algorithm.'
author = ["Vamsi"]
+++

I’m sure many of you must have read about and/or used Linear Regression, but people often remain unclear about it’s internal working.

So in this blog, I have four main goals:

- Derive the **normal equation** of LR in batch form. Along the way, I’ll go through a bit on how we tackle differentiation of vectors and matrices (matrix calculus) in the context of Machine Learning.

- **Justify the Least Squares Error** as the loss function for Linear Regression — why it makes sense mathematically and intuitively.

- **Explain the need for gradient descent** — why we use it and how it works.

- **Implement everything** we learn using python and numpy.

## Introduction

Let us start by introducing the problem. Suppose you have a housing dataset (classic example!) consisting of various features related to a house for example, number of bedrooms, proximity to school, house age etc. and also it’s price. Your task is to find the price of a house given a new set of features.

{{< figure src="california_housing_dataset.png" alt="California Housing Dataset" caption="<p style='text-align:center;'>California Housing Dataset</p>" >}}


Let's say we have $n$ such examples, meaning $n$ rows, and each row contains $d$ features (i.e., $d$ columns). We denote this by $X$.

The output is a **scalar value**, representing the price of the house. Since there are $n$ examples, we have $n$ such values, one for each row. We denote this by $y$.

Thus,

- The shape of $X$ is $(n,d)$.
- The shape of $y$ is $(n,1)$.

Now, let us assume linear relationship between the price of the house and the features of the house. We can express this relationship as a function. 

For a single example $x$ which is one single row of $X$,

$$
h_{\theta}(x^{(i)}) = \theta_{1} x_1^{(i)} + \theta_{2} x_2^{(i)} + \theta_{3} x_3^{(i)} + \dots + \theta_{d} x_d^{(i)}
$$

where:

- $h_θ(x^{(i)})$ is the price of the house for the ith row of X.
- $x_1^{(i)},x_2^{(i)},x_3^{(i)} \dots x_d^{(i)}$ are the **features of the house** (e.g., number of bedrooms, house age).
- $θ_1,θ_2,θ_3 \dots θ_d$​ are the **weights** (parameters) that determine how much each feature contributes to the price.

The above equation can also be written as,

$$
h_{\theta}(x^{(i)}) = x^{(i)} {\theta}
$$

- $x^{(i)}$ is of the shape $(1,d)$ and $θ$ is of the shape $(d,1)$ (you can see that the shapes are compatible and you can do matrix multiplication)

- $h_θ(x^{(i)})$ is of the shape $(1,1)$ which is nothing but a scalar and it is what you expect because it is the price of a single house.

Now, we can stack all the examples (or rows) together and write it in a compact matrix form like,

$$
\hat{y} = h_{\theta}(X) =  X \theta
$$

- $X$ is of the shape $(n,d)$ and $θ$ is of the shape $(d,1)$.

- $h_θ(X)$ and $\hat{y}$ are of the shape $(n,1)$ which are the prices of $n$ houses stored in a vector.

