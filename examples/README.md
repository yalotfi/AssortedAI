## Example: Kaggle's Digit Recognition

### Task:

Classify 28x28 resolution images of handwritten digits [0-9], doing so from scratch (NumPy)

### Description:

Kaggle's training set is distributed as a csv wherein each row represents a single image. As such, each image has been flattened to a `(1, 784)` vector but can be represented as a `(28, 28)` matrix. The first column of both the train and test files are the labels. These are real values between 0 and 255.

The models will be multi-class classifiers and so the predicted label, `y_hat`, and the actual label, `y`, will be represented as vectors of length `k_classes`; with 10 possible digits, these label vectors will have shape: `(1, 10)`. These vectors are stacked into a matrix of shape `(m_examples, k_classes)`.

Concretely, each label is a sparse (one-hot) vector meaning every element is zero except for the index of the actual class which is 1. Given an image of a 3, its one-hot encoded vector would be: `[0, 0, 0, 1, 0, 0, 0, 0, 0, 0]`. It is important to note that in this case, the 0th index corresponds to the class for 0 and the 10th index is the class for 9. It could be flipped, but I will assume this structure.

### Models to Build:

1. Softmax Regression (1-layer NN)
2. Shallow Neural Network (2-layer NN)
3. Deep Neural Network (L-layer NN)

Intuition for neural networks begins with logistic regression wherein they behave as the most basic feed-forward network possible. Data is passed through a linear function and the logistic (sigmoid) function squeezing the prediction to a real value between 0 and 1 (the prediction).

The negative log-liklihood function, or categorical-cross entropy, computes the error of the model's paramters. In order to minimize this loss, we update the weights and biases using stochastic gradient descent. The gradient is simply the sum of partial derivative of the loss function with respect to each parameter. This value is scaled by a set learning rate and adjusts the weights and biases. This update is done iteratively over a number of training iterations or "epochs."

All that really changes with deeper and wider feed-forward neural networks is their capability to represent more complex features. The fundamental concept remains the same wherein it makes a prediction given data and parameters, computes how far off it was given the "correct answer," and then adjusts parameters based on each ones contribution to the overall loss.

