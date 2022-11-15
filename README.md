# Accelerated-kmeans

In this repository, accelerated-kmeans and GMM-EM algorithms are implemented.

## Standard K-means
K-means clustering is one of the fundamental unsupervised learning algorithms. What we basically do in the basic K-means Clustering is to group a set of 
unlabeled data points into different clusters which can be applied to engineering applications like vector quantization.

The basic two steps are assignments and refitting. Usually, there will be several random points in the first setting which are centroids. However, 
random generation can yield poor performance, and the method we employ to tackle the issue is K-means++. K-means++, also known as farthest point clustering, 
picks the initial point uniformly at random. Then each subsequent point is picked from the remaining points with probability to its squared distance to
the closest cluster center. The figure below shows part of the implementation.

Once we have the centers, we assign each data to its closest center. Then, update the new center for each cluster with new data assigned to it. The two steps are 
iteratively performed until convergence.

## Accelerated K-means
When applying standard k-means, we need to calculate the distance between data points and each center. This is time-consuming and yields $O(n \cdot K \cdot d)$.

Motivated by triangle inequality, researchers have discovered a more time-efficient way of solving k-means clustering which avoid redundant calculation and 
make use of the previous results (On the other hand, this is space-consuming). The lemmas derived are stated as follows.

- Lemma 1: Let $x$ be a point and let $b$ and $c$ be centers. If $d(b,c) \geq 2d(x,b)$, then $d(x,c) \geq d(x,b)$.

- Lemma 2: Let $x$ be a point and let $b$ and $c$ be centers. Then $d(x,c) \geq max$ $(0, d(x,b)-d(b,c))$

## GMM-EM

Gaussian mixture model aims to predict the latent Gaussian distributions that contribute to the current observations. One well-known algorithm to achieve 
this goal is apply Expectation-Maximization algorithm which highly resembles K-means clustering.

Given means, covariances and mixing coefficients, we first perform the expectation step. Repeat the above process until convergence. For loss function,
we employ log-likelihood to estimate the model.
