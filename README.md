# DeepMiRBP: A Hybrid Model for Predicting MicroRNA-Protein Interactions

## Overview
DeepMiRBP is a novel hybrid deep learning model designed to predict microRNA-binding proteins by modeling molecular interactions. The model integrates Bidirectional Long Short-Term Memory (Bi-LSTM) networks, Convolutional Neural Networks (CNNs), and cosine similarity to offer a robust computational approach for inferring microRNA-protein interactions.

## Table of Contents
- [Overview](#overview)
- [Dependencies](#Dependencies)
- [Usage](#usage)
- [Model Components](#model-components)
- [License](#license)



## Overview

DeepMiRBP is an advanced hybrid deep learning model that predicts interactions between microRNAs (miRNAs) and RNA-binding proteins (RBPs). The model leverages the power of Bidirectional Long Short-Term Memory (Bi-LSTM) networks, Convolutional Neural Networks (CNNs), and cosine similarity to provide accurate predictions of miRNA-protein interactions. 

### Key Features
- **Hybrid Model Architecture**: Combines Bi-LSTM networks for sequence encoding and CNNs for structural data processing, enhancing prediction accuracy.
- **Attention Mechanism**: Focuses on the most relevant parts of the RNA sequences to improve interaction prediction.
- **Transfer Learning**: Utilizes knowledge transfer from a pre-trained source domain to target domain tasks, increasing model efficiency and effectiveness.
- **Cosine Similarity**: Measures the similarity between miRNAs and proteins to identify potential binding interactions.
- **Comprehensive Evaluation**: Includes detailed performance metrics and validation through case studies to demonstrate model robustness.

### Applications
DeepMiRBP can be applied in various fields of computational biology and bioinformatics, including:
- **Gene Regulation Studies**: Understanding how miRNAs regulate gene expression by binding to specific RBPs.
- **Disease Mechanism Exploration**: Investigating the role of miRNA-protein interactions in diseases like cancer.
- **Therapeutic Target Identification**: Identifying novel miRNA-protein interactions that can be targeted for therapeutic interventions.

This repository provides the necessary scripts and instructions to train the DeepMiRBP model, generate embeddings, and predict miRNA-protein interactions. The model's architecture and approach offer researchers a robust tool for exploring and understanding the complex interplay between miRNAs and RBPs.

## Dependencies

To ensure the proper functioning of DeepMiRBP, please make sure you have the following dependencies installed:

- **Python**: 3.8 or higher
- **NumPy**: v1.19.5
- **SciPy**: v1.6.2
- **Keras**: v2.4.3 (backend: TensorFlow v2.4.1)
- **Scikit-learn**: v0.24.1
- **RNAshapes**: If you cannot download it, please use the provided RNAshapes in this GitHub repository.

### Installation Instructions

1. **Clone the Repository**:
    ```sh
    git clone https://github.com/yourusername/DeepMiRBP.git
    cd DeepMiRBP
    ```

2. **Install the Required Packages**:
    ```sh
    pip install -r requirements.txt
    ```


3. **Install RNAshapes**:
    - If RNAshapes is not available for download, use the provided RNAshapes file in the repository:
        ```sh
        cd RNAshapes
        chmod +x RNAshapes
        ```

These steps ensure that all necessary dependencies are installed and configured correctly for running the DeepMiRBP model.



## Model Components

The DeepMiRBP model is divided into two main components: the first part focuses on identifying candidate RNA-binding proteins (RBPs) using transfer learning and cosine similarity. In contrast, the second part refines these candidates using protein structural information.

### First Component: Identifying Candidate RBPs

1. **Source Domain**:
   - The source domain is trained on known RNA sequences that bind to RBPs.
   - It leverages transfer learning to apply the knowledge gained from RNA-RBP interactions to miRNA sequences.
   - Embedding layers convert sequences into vectors.
   - Cosine similarity is used to find RNA sequences most similar to the miRNA.

2. **Target Domain**:
   - The target domain receives miRNA sequences as input.
   - It uses the learned embeddings and cosine similarity to generate a list of candidate RBPs.

### Second Component: Refining Candidate RBPs

1. **PSSM and Contact Map Processing**:
   Position-specific scoring Matrices (PSSMs) and contact maps of proteins capture evolutionary and structural information.
   - Encoder-decoder networks process these matrices to create detailed protein representations.

2. **Cosine Similarity for Protein Comparison**:
   - Cosine similarity is applied to compare proteins based on their structural representations.
   - The output is an \(n \times n\) matrix indicating the similarity between proteins, helping identify those most likely to bind to the miRNA.

Below is a visual representation of the model components:

![Model Components](path/to/Overview-model2.png)

This figure illustrates the workflow of the DeepMiRBP model, showing the two-part architecture and the interaction between the source and target domains.

![Overview-model2](https://github.com/sbbi-unl/DeepmiRBP/assets/55287271/1e7849b1-945b-48ff-867d-b8e8b2f082a7)


##License
This content should be easy to follow and suitable for a README.md file in a GitHub repository. It provides an overview of the model, installation instructions, and a high-level description of the model components and their usage.

