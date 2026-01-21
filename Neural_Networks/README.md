# Neural Network from Scratch for XOR Problem

## Project Description
This project demonstrates the implementation of a simple neural network from scratch using NumPy to solve the classic XOR problem. The XOR (exclusive OR) problem is a fundamental challenge in neural networks because it is not linearly separable. A single-layer perceptron cannot solve it, as it can only classify linearly separable patterns. However, a multi-layer neural network with a hidden layer can learn to approximate the non-linear decision boundary required for XOR, showcasing the power of hidden layers in learning complex patterns.

## Neural Network Implementation
This notebook implements a two-layer neural network (one hidden layer) to tackle the XOR problem. The architecture and core processes are as follows:

*   **Input Layer**: Consists of 2 neurons, corresponding to the two input features of the XOR problem (e.g., [0, 0], [0, 1], [1, 0], [1, 1]).
*   **Hidden Layer**: Contains 2 neurons, which allow the network to learn non-linear relationships between the inputs and outputs. A Sigmoid activation function is applied here.
*   **Output Layer**: Comprises 1 neuron, producing the final output (0 or 1) for the XOR operation. A Sigmoid activation function is also used for the output.
*   **Activation Function**: The Sigmoid function `(1 / (1 + exp(-z)))` is used throughout the network. Its derivative `(a * (1 - a))` is crucial for backpropagation.
*   **Forward Propagation**: Inputs are passed through the network, layer by layer, with weighted sums and activation functions applied at each step, to produce an output prediction.
    *   `z1 = X * weights1 + bias1`
    *   `a1 = Sigmoid(z1)` (Hidden layer output)
    *   `z2 = a1 * weights2 + bias2`
    *   `a2 = Sigmoid(z2)` (Output prediction)
*   **Backpropagation**: The error between the predicted and actual output is used to calculate gradients (derivatives of the loss with respect to weights and biases) and propagate them backward through the network.
    *   `delta2 = (a2 - Y) * Sigmoid_derivative(a2)`
    *   `delta1 = delta2 * weights2.T * Sigmoid_derivative(a1)`
*   **Weight and Bias Updates**: Weights and biases are adjusted iteratively using gradient descent to minimize the loss function (Mean Squared Error).
    *   `weights -= learning_rate * gradients`
    *   `bias -= learning_rate * gradients`

## How to Set Up and Run the Code
This notebook is designed to be run in Google Colaboratory. Follow these steps:

1.  **Open the Notebook**: Access the `.ipynb` file in Google Colab.
2.  **Ensure Runtime**: Make sure the runtime type is set to Python 3 (default).
3.  **Run Cells Sequentially**: Execute each code cell in order, from top to bottom.
    *   The initial cells import necessary libraries (NumPy, Matplotlib), prepare the XOR dataset, define hyperparameters, and initialize weights/biases.
    *   The `Sigmoid` activation functions are defined.
    *   The main training loop will execute for `epochs` (e.g., 10,000 iterations), performing forward and backpropagation, and updating weights.
    *   The loss curve will be plotted.
    *   Finally, the model will make predictions on the training data.

## Expected Results
Upon running the notebook, you should observe the following:

*   **Decreasing Loss**: The training loss (Mean Squared Error) will steadily decrease over epochs, indicating that the neural network is learning and converging. The plot of training loss versus epochs will show a downward trend.
*   **Accurate Predictions**: After training, the model should accurately predict the XOR outputs for the given inputs:
    *   Input [0, 0] -> Prediction [0]
    *   Input [0, 1] -> Prediction [1]
    *   Input [1, 0] -> Prediction [1]
    *   Input [1, 1] -> Prediction [0]

This demonstrates that the neural network has successfully learned the non-linear XOR logic.