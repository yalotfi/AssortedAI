## AssortedAI

This project is primarily a way to organize individual ML algorithms that I've implemented in disparate repositories. AssortedAI will become a package that has core functionality that makes machine learning simple and extensible. At a bare minimum, it will cover basic supervised learning algorithms, at least one clustering technique, variations of stochastic gradient descent, and some preprocessing utilities.

Beyond the basic prerequisites, I would like the implement visualization tools and dimensionality reduction techniques like PCA and t-SNE. In terms of graphing model performance, many libaries are lacking in this regard. This feature could become an advantage. In terms of optimization, I haven't used second order, Netown methods so this could be another stretch goal for the project.

Finally, there will be several fleshed out examples with concrete machine learning tasks from MNIST to iris-flower prediction, and housing price predictions. It will provide a solid context on using the APIs and I aim for it to feel akin to a simplified sklearn work-flow.

### Motivation:

1. My GitHub account has become pretty *disorganized with many half-baked projects.* The structure and focus of building a dedicated package will help clean these repos.
2. *Education!* I have found that for myself, implementing these algorithms from scratch provides a much stronger intuition for how they work.
3. *Simplicity.* Many newcomers to ML, like myself, find a lot of the most popular packages to be very overwhelming. Given that this package is meant to be an educational experiment, hopefully it will prove the same for others.

### Models:

1. Linear Regression
2. Softmax Regression (generalized logistic regression)
3. Shallow (2-L) Feedforward Neural Networks
4. Deep (L-layer) Feedforward Neural Networks
5. K-Nearest Neighbors
6. K-Means Clustering

### Utilities/Other:

1. Activation Functions
    * Sigmoid
    * Hyperbolic Tangent
    * ReLU
    * Leaky-ReLU
2. Optimization Methods
    * Stochastic Gradient Descent
    * Adam
    * RMSProp
3. Preprocessing Tools
    * One-hot encoding
    * Normalization
