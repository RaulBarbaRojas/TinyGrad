# 🚀 TinyGrad

> A tiny, elegant automatic differentiation engine built from scratch.

TinyGrad is a lightweight module for **automatic gradient computation**, the core mechanism behind training neural networks using **backpropagation**.

Inspired by:
- 🔬 [MicroGrad](https://github.com/karpathy/micrograd)
- 🔥 [PyTorch Autograd](https://docs.pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

...but extended to support more features than MicroGrad, without all the complexity of PyTorch. This module supports building **complete neural networks** and has a **clear educational purpose**.

---

## ✨ Features

- 🧠 Reverse-mode automatic differentiation (backpropagation)
- 🔢 Scalar-based computational graph
- 🔄 Dynamic graph construction
- 🏗️ Build full neural networks (Linear layer support for now! 🗓️)
- 🪶 Lightweight and easy to read
- 📚 Fully Educational

---

## 📦 Installation

You can run the following command to install `tinygrad` in your system/virtual environment:

`pip install tinygrad@git+https://github.com/RaulBarbaRojas/TinyGrad.git`

## 🧩 Example Usage

The usage of `tinygrad` to build expressions and run backpropagation (automatic differentation) is shown next:

```python
from tinygrad.core import Parameter

a = Parameter(2.0)
b = Parameter(0.3)
c = Parameter(1.0)

ab = a * b
f = ab + c
print(f'f={f.value:.4f}')
f.backward()
print(f'grad_f={f.grad:.4f}')
print(f'grad_a={a.grad:.4f}')
print(f'grad_b={b.grad:.4f}')
print(f'grad_c={c.grad:.4f}')
```

An example training loop of tinygrad is given next:

```python
from tinygrad.nn.linear import Linear
from tinygrad.nn.activations import ReLU
from tinygrad.nn.loss.mse import MSE
from tinygrad.optimizer.sgd import SGD

X = [2.00, 2.10, 2.20, 2.30, 2.40]
y = [3.00, 3.10, 3.13, 3.16, 3.19]

epochs = 2000
model = Linear(input_features=1, output_features=1, activation_fn=ReLU())
model.neurons[0].weights[0].value = 0.5  # Using positive weight to prevent Dying ReLU
optimizer = SGD(model.parameters(), lr=1e-2)
loss_fn = MSE()

for _ in range(epochs):
    for x_value, y_true in zip(X, y):
        optimizer.zero_grad()
        y_pred = model([x_value])
        loss = loss_fn(y_pred, [y_true])
        loss.backward()
        optimizer.step()
        print(f'Loss: {loss.value:.4f}')
```
