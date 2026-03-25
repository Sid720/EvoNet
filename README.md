# EvoNet
EvoNet is a self-evolving neural network project built to demonstrate how a model can change its own architecture during training. Instead of staying fixed from the start, the network can expand in width or depth when learning progress slows down, making the training process adaptive and dynamic.

## Project Overview
Traditional neural networks are usually designed first and trained later. In EvoNet, the model begins with a simple structure and grows when needed. This project focuses on showing that idea in a clean and understandable way using PyTorch and Streamlit.

The application provides a live interface where users can control training settings, observe loss behavior, monitor parameter growth, and view mutation events as the network evolves.

## Features
- Self-evolving neural network architecture
- Width mutation to increase neurons in a hidden layer
- Depth mutation to add a new hidden layer
- Synthetic nonlinear regression dataset generated at runtime
- Live Streamlit dashboard for training and mutation tracking
- Real-time visualization of:
  - training loss
  - validation loss
  - parameter growth
  - architecture updates
- Reset option to restart the model from its initial state

## Tech Stack

- Python
- PyTorch
- Streamlit
- NumPy
- Pandas

How EvoNet Works
The model starts with a small neural network architecture.
A synthetic dataset is generated for a nonlinear regression task.
During training, the model tracks validation loss.
If improvement slows down for a sustained period, the model mutates.
Mutation can happen in two ways:
Width mutation: adds more neurons to an existing hidden layer
Depth mutation: adds an additional hidden layer
Training continues with the new structure.
This approach makes the model more flexible and demonstrates the concept of adaptive architecture design.

## Live Demo


## Why This Project:
EvoNet is useful as a learning project for understanding:
1.neural network architecture design
2.dynamic model growth
3.mutation-based adaptation
4.interactive ML visualization with Streamlit
5.It is not intended as a production-grade adaptive learning system, but as a practical and educational 6.demonstration of self-evolving neural networks.

## Future Improvements:
save model checkpoints after mutations
compare multiple training runs
support real datasets in addition to synthetic data
add exportable training logs
visualize architecture changes more clearly over time