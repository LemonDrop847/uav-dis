# ChainFed

## Description

ChainFed is a pioneering framework that integrates blockchain technology with Federated Learning to create a secure and transparent collaborative machine learning ecosystem. It addresses key challenges like trust, transparency, and fair incentivization in Federated Learning.

ChainFed uses smart contracts to manage contribution scores, reward allocation, and verify client updates hashed with SHA256. The framework incorporates differential privacy to enhance data security and supports diverse data scenarios, including IID and non-IID distributions. Simulations on MNIST and CIFAR-10 datasets demonstrate ChainFed's ability to maintain model accuracy, scalability, and fairness, making it a robust solution for decentralized learning.

## Features

- **Decentralized Learning:** Utilizes Federated Learning for decentralized model training on client devices.
- **Blockchain Integration:** Ensures secure and transparent update management with blockchain-based smart contracts.
- **Privacy Protection:** Incorporates differential privacy techniques for enhanced data confidentiality.
- **Flexible Data Scenarios:** Supports both IID and non-IID data distributions, catering to diverse data conditions.
- **Incentivization Mechanisms:** Implements fair reward allocation to encourage client collaboration.
- **Comprehensive Simulations:** Evaluated on MNIST and CIFAR-10 datasets with competitive accuracy and scalability.

## Getting Started

### Prerequisites

- Solidity development environment (e.g., Hardhat or Foundry)
- Node.js and npm
- Python (for simulation and evaluation scripts)
- Dependencies for running simulations (e.g., TensorFlow or PyTorch)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/LemonDrop847/ChainFed.git
   cd chainFed
   ```

2. Install dependencies:

   ```bash
   cd blockchain_main
   make all
   ```

3. Run anvil:

   ```bash
   anvil
   ```

4. Deploy smart contracts:

   ```bash
   make deploy
   ```

5. Run simulations:

   ```bash
   cd fl_main/with_block
   pip install requirements.txt
   python run_simulation.py
   ```

## License

This project is licensed under the MIT License. See the LICENSE file for details.

