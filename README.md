# Deep Learning with Python

Deep learning is a subfield of machine learning that uses artificial neural networks with multiple layers to learn representations of data with increasing levels of abstraction. Inspired by the structure of the human brain, deep learning models can automatically discover the features needed for detection or classification directly from raw data, such as images, text, or audio, without the need for manual feature engineering.

Deep learning has driven significant advances in computer vision, natural language processing, speech recognition, and generative modeling. Python has become the dominant language for deep learning research and practice, supported by powerful open-source frameworks and a rich scientific computing ecosystem.

The foundations of deep learning are covered in detail in the textbook [Deep Learning](https://www.deeplearningbook.org/) by Ian Goodfellow, Yoshua Bengio, and Aaron Courville (MIT Press, 2016), which spans applied mathematics, practical deep network design, and research-level topics. A practitioner-oriented companion is [Deep Learning with Python](https://deeplearningwithpython.io/chapters/) by Francois Chollet, covering topics from mathematical building blocks through image classification, natural language processing, and generative models using TensorFlow and Keras.

---

## Repository Structure

```
📁 Deep-Learning-with-Python/
├── 📁 Components-of-Machine-Learning/
├── 📁 Artificial-Neural-Networks/
├── 📁 Gradient-Based-Learning/
├── 📁 Convolutional-Neural-Nets/
├── 📁 Regularization/
├── 📁 Natural-Language-Processing/
├── 📁 Generative-Adversarial-Networks/
└── 📁 coursedata/
```

| Folder | Topic |
|---|---|
| [Components-of-Machine-Learning](./Components-of-Machine-Learning) | Python basics, Pandas DataFrames, data loading, and core machine learning concepts |
| [Artificial-Neural-Networks](./Artificial-Neural-Networks) | ANN architecture, activation functions, regression and classification with Keras |
| [Gradient-Based-Learning](./Gradient-Based-Learning) | Gradient descent, stochastic gradient descent, loss functions, and optimization |
| [Convolutional-Neural-Nets](./Convolutional-Neural-Nets) | Convolutional layers, pooling, and image classification with CNNs |
| [Regularization](./Regularization) | Overfitting prevention, data pipelines, data augmentation, and transfer learning |
| [Natural-Language-Processing](./Natural-Language-Processing) | Text classification, bag-of-words, word embeddings, and sequence models |
| [Generative-Adversarial-Networks](./Generative-Adversarial-Networks) | GAN architecture, generator and discriminator training |
| [coursedata](./coursedata) | Datasets used across exercises (e.g., cats and dogs image sets) |

---

## Python Libraries for Deep Learning

Three libraries form the core of the Python deep learning ecosystem used in this repository:

- **TensorFlow / Keras** — A high-level API and ecosystem for building, training, and deploying models. Keras provides a clean, modular interface on top of TensorFlow's computation graph.
- **PyTorch** — Favored in research for its dynamic computation graph and intuitive, Pythonic interface. Eager execution makes debugging and experimentation straightforward.
- **Scikit-learn** — Useful for data preprocessing, model evaluation, and traditional machine learning algorithms that often serve as baselines against which deep models are compared.

---

## Neural Networks

A neural network is a composition of alternating affine mappings (defined by weights and biases) and non-linear activation functions. This structure allows the network to approximate arbitrarily complex functions given sufficient depth and width.

Deep learning models consist of layers of interconnected neurons:

- **Input Layer** — Receives raw data such as images, text, or numerical feature vectors.
- **Hidden Layers** — Where feature extraction occurs. Each successive layer learns increasingly abstract representations. Deep networks contain multiple hidden layers; Convolutional Neural Networks (CNNs) are suited to spatial data such as images, while Recurrent Neural Networks (RNNs) and Transformers process sequential data.
- **Output Layer** — Produces the final prediction, such as a classification label or a continuous value.

A single artificial unit computes a weighted sum of its inputs plus a bias term:

```
z = b + w1*x1 + w2*x2 + ... + wn*xn
```

The unit then applies a non-linear activation function `g` to produce its output, called the activation:

```
output = g(z)
```

---

## Activation Functions

Activation functions introduce non-linearity into the network, which is essential for learning complex patterns. Without non-linearity, a deep network would collapse to a single linear transformation regardless of depth.

| Activation | Formula | Common Use |
|---|---|---|
| ReLU | `g(z) = max(0, z)` | Default choice for hidden layers in most architectures |
| Sigmoid | `g(z) = 1 / (1 + e^(-z))` | Output layer for binary classification |
| Softmax | `g(z_i) = e^(z_i) / sum(e^(z_j))` | Output layer for multi-class classification |
| Tanh | `g(z) = (e^z - e^(-z)) / (e^z + e^(-z))` | Hidden layers in RNNs; outputs are in the range (-1, 1) |

**ReLU (Rectified Linear Unit)** is the most widely used activation function for hidden layers. It outputs the input directly if positive, and zero otherwise:

$$g(z) = \max(0, z)$$

**Sigmoid** squashes any real-valued input into the interval (0, 1), making it suitable for binary output probabilities:

$$g(z) = \frac{1}{1 + e^{-z}}$$

**Tanh** is a rescaled sigmoid that squashes values into (-1, 1) and is frequently used in recurrent architectures:

$$g(z) = \frac{e^{z} - e^{-z}}{e^{z} + e^{-z}}$$

---

## Optimization

Training a neural network means finding the parameter values (weights and biases) that minimize a loss function over the training data. This is achieved through backpropagation combined with an iterative optimization algorithm.

**Loss Functions**

For a predicted label $\hat{y}$ and a true label $y$, the loss function $L(y, \hat{y})$ measures the prediction error. Common choices are:

- Squared error loss for regression: $L(y, \hat{y}) = (y - \hat{y})^2$
- Cross-entropy loss for classification

**Gradient Descent**

Gradient Descent (GD) minimizes the training error by repeatedly moving the parameter vector in the direction of the negative gradient of the loss:

$$\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \, \nabla_{\mathbf{w}} f(\mathbf{w}^{(k)})$$

where $\alpha$ is the learning rate and $f(\mathbf{w})$ is the average loss (cost function) over the training set.

**Stochastic Gradient Descent (SGD)**

Rather than computing the gradient over the entire dataset, SGD updates the weights using the gradient computed from a small random mini-batch of training samples. This makes each update computationally cheap and introduces beneficial noise that can help escape shallow local minima.

**Adam (Adaptive Moment Estimation)**

Adam is a widely used optimizer that maintains per-parameter adaptive learning rates by combining the benefits of AdaGrad (which scales learning rates by historical gradient magnitudes) and RMSProp (which uses an exponentially decaying average of squared gradients). Adam typically converges faster than plain SGD on most deep learning tasks.

**Regularization**

Regularization techniques reduce overfitting by constraining the model's complexity:

- **Dropout** — During training, randomly sets a fraction of neuron activations to zero, forcing the network to learn redundant representations and preventing co-adaptation of neurons.
- **L1 / L2 Regularization** — Adds a penalty term to the loss function proportional to the absolute values (L1) or squared values (L2) of the weights, discouraging overly large weights.
- **Data Augmentation** — Artificially increases the diversity of training data through transformations (flips, crops, rotations) to improve generalization.
- **Transfer Learning** — Reuses weights from a model pre-trained on a large dataset, adapting only the top layers to the target task, which is particularly effective when labeled data is scarce.

---

## Autoencoders

An autoencoder is a neural network trained to copy its input to its output. Internally it has a hidden layer $\mathbf{h}$ that describes a compressed code used to represent the input. The network consists of two parts: an encoder function $\mathbf{h} = f(\mathbf{x})$ that maps the input to a latent representation, and a decoder function $r = g(\mathbf{h})$ that reconstructs the input from that representation.

The learning objective forces the model to discover a compact, informative representation of the data — only the most salient structure is preserved in the bottleneck. Autoencoders are applied to dimensionality reduction, anomaly detection, and as components of more complex generative models.

For a full treatment, see [Chapter 14 — Autoencoders](https://www.deeplearningbook.org/contents/autoencoders.html) in the Deep Learning textbook.

---

## References

- Ian Goodfellow, Yoshua Bengio, and Aaron Courville. *Deep Learning*. MIT Press, 2016. [https://www.deeplearningbook.org/](https://www.deeplearningbook.org/)
- Francois Chollet. *Deep Learning with Python*. Manning Publications. [https://deeplearningwithpython.io/chapters/](https://deeplearningwithpython.io/chapters/)
- Chapter 14: Autoencoders. [https://www.deeplearningbook.org/contents/autoencoders.html](https://www.deeplearningbook.org/contents/autoencoders.html)
