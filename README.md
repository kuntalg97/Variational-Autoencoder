# Variational-Autoencoder

A small illustration of generative AI for computational chemistry: a **Variational Autoencoder (VAE)** that learns from a handful of molecular structures along a reaction pathway and generates thousands of new, closely related structures.

## Table of Contents

- [Overview](#overview)
- [Motivation](#motivation)
- [What Is a Variational Autoencoder?](#what-is-a-variational-autoencoder)
- [The Chemical System](#the-chemical-system)
- [Repository Contents](#repository-contents)
- [How the Code Works](#how-the-code-works)
- [Model Architecture](#model-architecture)
- [Training Details](#training-details)
- [Installation](#installation)
- [Usage](#usage)
- [Output](#output)
- [Customizing the Code](#customizing-the-code)
- [Future Directions](#future-directions)
- [Author](#author)

## Overview

This code is a small illustration of generative AI, specifically Variational Autoencoders (VAEs) ([IBM: What is a variational autoencoder?](https://www.ibm.com/think/topics/variational-autoencoder)). VAEs are deep learning models that encode input data into a continuous, probabilistic latent space. After optimizing a loss function, they can generate new data that are small variations of the original dataset. This makes VAEs powerful tools for generating new samples that very closely resemble the original data.

In this project, a VAE is trained on **71 configurations** of an S<sub>N</sub>2 reaction and is then used to generate **10,000 new configurations** along the same reaction coordinate.

## Motivation

In quantum chemical calculations, an extensive sampling of the potential energy surface (PES) of molecular systems is critically important. However, such quantum mechanical (QM) calculations are typically expensive.

The original dataset here consists of ~70 configurations of an S<sub>N</sub>2 reaction along a reaction coordinate. It was constructed from a combination of shell and Tcl scripting, followed by QM minimization using [CP2K](https://www.cp2k.org/). Constructing ~10,000 structures along the same reaction coordinate with these methods would be non-trivial and expensive.

This VAE code accomplishes the same goal by encoding the original set of ~70 structures and, by training on just the atomic coordinates, generating ~10,000 structures at a fraction of the computational cost.

## What Is a Variational Autoencoder?

A VAE has two main parts:

1. **Encoder** – compresses each input (here, a molecular structure) into a small set of numbers called the *latent space*. Instead of a single point, the encoder outputs a probability distribution, described by a mean (μ) and a variance (σ²).
2. **Decoder** – takes a point sampled from the latent space and reconstructs a full molecular structure from it.

During training, the model learns two things at once:

- **Reconstruction**: structures that go through the encoder and decoder should come out looking like the originals.
- **Regularization (KL divergence)**: the latent space should stay smooth and close to a standard normal distribution, so that any random point drawn from it decodes into a realistic structure.

Once trained, new structures are created by simply drawing random points from the latent space and passing them through the decoder.

```
 Original structure ──► Encoder ──► (μ, σ) ──► sample z ──► Decoder ──► Reconstructed structure
                                                   ▲
                       Generation:  random z ──────┘  (+ reaction coordinate)
```

### Conditioning on the reaction coordinate

This implementation is a **conditional VAE**: the decoder receives both the latent vector *z* and a **reaction coordinate** value between 0 and 1. This allows the user to ask the model for structures at any chosen point along the reaction pathway — from reactants (0) to products (1) — and to fill in the pathway densely.

## The Chemical System

The system is the classic gas-phase S<sub>N</sub>2 reaction between a chloride ion and chloromethane:

```
Cl⁻ + CH₃Cl  ──►  ClCH₃ + Cl⁻
```

Each configuration contains **6 atoms**: one carbon (C), three hydrogens (H), and two chlorines (Cl). Along the reaction coordinate, one C–Cl bond breaks while the other forms, passing through the characteristic trigonal-bipyramidal transition state with the methyl group inverting ("umbrella flip").

## Repository Contents

| File | Description |
|------|-------------|
| `generate_vae.py` | Main Python script: reads the data, builds and trains the VAE, generates new structures, and plots the results. |
| `original_configurations.xyz` | Training data: 71 QM-minimized structures (CP2K) along the S<sub>N</sub>2 reaction coordinate. |
| `new_configurations_vae.xyz` | Example output: 10,000 VAE-generated structures along the same reaction coordinate. |
| `README.md` | This file. |

All structure files use the standard [XYZ format](https://en.wikipedia.org/wiki/XYZ_file_format): the number of atoms, a comment line, and then one line per atom containing the element symbol and its x, y, z coordinates (in Å).

## How the Code Works

`generate_vae.py` runs through the following steps:

1. **Load data** – The original structures are read with [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment). Each structure's 6 × 3 atomic coordinates are flattened into a vector of 18 numbers.
2. **Assign reaction coordinates** – Each structure is labeled with a reaction coordinate evenly spaced between 0 and 1, following the order of the frames in the file.
3. **Prepare the dataset** – Coordinates and reaction-coordinate labels are combined into PyTorch tensors and served in shuffled mini-batches of 16.
4. **Build the VAE** – An encoder, a latent space, and a decoder conditioned on the reaction coordinate (see [Model Architecture](#model-architecture)).
5. **Train** – The model is trained for 10,000 epochs using the Adam optimizer, with the loss printed every 100 epochs.
6. **Generate** – 10,000 reaction-coordinate values are spaced evenly between 0 and 1. For each one, a random latent vector is drawn from a standard normal distribution and decoded into a new structure.
7. **Save** – The generated structures are written to `new_configurations_vae.xyz`, using the same atom ordering and element types as the original data.
8. **Visualize** – A scatter plot compares the x-coordinate of the first atom (carbon) in the original and generated structures as a function of the reaction coordinate.

## Model Architecture

```
Encoder:   18 coordinates ─► Linear(128) ─► ReLU ─► Linear(64) ─► ReLU ─┬─► μ       (16)
                                                                         └─► log σ² (16)

Latent:    z = μ + σ · ε,   ε ~ N(0, I)                     (reparameterization trick)

Decoder:   [z (16) + reaction coordinate (1)] ─► Linear(64) ─► ReLU ─► Linear(128) ─► ReLU ─► Linear(18)
```

| Component | Details |
|-----------|---------|
| Input size | 18 (6 atoms × 3 Cartesian coordinates) |
| Hidden layers | 128 and 64 units, ReLU activation |
| Latent dimension | 16 |
| Decoder condition | Reaction coordinate (1 value in [0, 1]) |
| Output size | 18 (reconstructed coordinates) |

The **reparameterization trick** (z = μ + σ · ε) lets the model sample randomly from the latent space while still allowing gradients to flow back through the network during training.

## Training Details

| Setting | Value |
|---------|-------|
| Optimizer | Adam |
| Learning rate | 1 × 10⁻³ |
| Batch size | 16 |
| Epochs | 10,000 |
| Reconstruction loss | Mean squared error (MSE) |
| Regularization | KL divergence, weighted by 0.1 |

The total loss is:

```
Loss = MSE(reconstructed, original) + 0.1 × KL divergence
```

The factor of 0.1 balances the two goals: a small weight keeps the generated structures accurate, while still keeping the latent space smooth enough for sampling.

## Installation

The code requires Python 3 and the following packages:

- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [ASE](https://wiki.fysik.dtu.dk/ase/)
- [Matplotlib](https://matplotlib.org/)

Install them with `pip`:

```bash
pip install torch numpy ase matplotlib
```

or with `conda`:

```bash
conda install -c conda-forge pytorch numpy ase matplotlib
```

## Usage

Clone the repository and run the script from inside it:

```bash
git clone https://github.com/kuntalg97/Variational-Autoencoder.git
cd Variational-Autoencoder
python generate_vae.py
```

While running, the script prints:

- the shapes of the input tensors, and
- the training loss every 100 epochs, e.g. `Epoch [100/10000], Loss: ...`

Training runs comfortably on a standard CPU, since the dataset and the network are both small.

## Output

After training, the script produces:

1. **`new_configurations_vae.xyz`** – 10,000 newly generated structures ordered along the reaction coordinate from 0 to 1.
2. **A comparison plot** – the carbon atom's x-coordinate versus the reaction coordinate, with the original structures in red and the generated structures in blue. This gives a quick visual check that the generated structures follow the same trend as the training data.

The generated structures can be viewed with any molecular visualization program that reads XYZ files, for example:

- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
- [OVITO](https://www.ovito.org/)
- ASE's built-in viewer: `ase gui new_configurations_vae.xyz`

## Customizing the Code

The main settings are defined directly in `generate_vae.py` and are easy to change:

| Variable | Default | Meaning |
|----------|---------|---------|
| `latent_dim` | 16 | Size of the latent space |
| `num_epochs` | 10000 | Number of training epochs |
| `batch_size` (in `DataLoader`) | 16 | Number of structures per training batch |
| `lr` (in `optim.Adam`) | 1e-3 | Learning rate |
| `0.1` (in the loss) | 0.1 | Weight of the KL-divergence term |
| `num_new_configs` | 10000 | Number of structures to generate |

To apply the code to a different molecular system, replace `original_configurations.xyz` with your own XYZ trajectory. The frames should be ordered along the reaction pathway and should all contain the same atoms in the same order. The input and output sizes of the network adapt automatically to the number of atoms.
