## Kaggle MNIST Digit Recognition

### Task:

Classify 28x28 resolution images of handwritten digits [0-9] and to do so from scratch (NumPy)

### Description:

Kaggle's training set is distributed as a csv wherein each row represents a single image. As such, each image would have initially been represented as a `(28, 28)` matrix but has been flattened into a `(1, 784)` vector. The first column of both the train and test files are the labels.

By definition, the models will be multi-class classifiers and so the predicted label, `y_hat`, and the actual label, `y`, will be represented as vectors of length `n_classes`; with 10 possible digits, these label vectors will have shape: `(1, 10)`. The labels, `y`, will have to be converted into one-hot vectors meaning the element index of the actual class equals 1 and the rest are 0. Given an image of a 3, its one-hot encoded vector would be: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. It is important to note that in this case, the 0th index corresponds to the class for 0 and the 10th index is the class for 9. It could be flipped, but I will assume this structure.

### Models to Build:

1. Logistic Regression (1-layer NN)
2. Shallow Neural Network (2-layer NN)
3. Deep Neural Network (L-layer NN)

Intuition for neural networks begin with logistic regression wherein they behave as a single, feed-forward perceptron. All logistic regression is a linear function passed through a logistic function. This provides some non-linearity to the model as opposed to linear regression. Specifically, the logistic function or sigmoid, has the convenient characteristic of outputting a value between 0 and 1 thus providing a way to pull the most probable label given the current parameters.

To compute the cost, the negative log-loss function, or categorical-cross entropy, is also used. Finally, we update the weights and biases using stochastic gradient descent. The partial derivative of the loss with respect to each parameter, scaled by a set learning rate, adjusts the weights and biases. Again, calculate the next prediction, its error, and adjust accordingly.

All that really changes with fully connected, feed-forward neural networks is their capability to represent more complexity. The fundamental concept remains the same wherein it makes a prediction given data and parameters, computes how far off it was given the "correct answer," and then adjusts parameters based on each ones contribution to the overall loss.

### Dependencies:

1. `numpy` - optimized linear algebra
2. `pandas.read_csv()` - easily read in tabular data
3. `matplotlib` - plotting utilities
