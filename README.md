# Variational-Autoencoder

A small illustration of generative AI for computational chemistry: a **Variational Autoencoder (VAE)** that learns from a handful of molecular structures along a reaction pathway and generates thousands of new, closely related structures.

## Quick Start

If you just want to see it run:

```bash
git clone https://github.com/kuntalg97/Variational-Autoencoder.git
cd Variational-Autoencoder
pip install torch numpy ase matplotlib
python generate_vae.py
```

This trains the VAE on `original_configurations.xyz`, writes 10,000 new structures to `new_configurations_vae.xyz`, and opens a plot comparing the original and generated structures. The rest of this README explains each step in detail.

## Table of Contents

- [Quick Start](#quick-start)
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
- [Analyzing the Generated Structures](#analyzing-the-generated-structures)
- [Customizing the Code](#customizing-the-code)
- [Using Your Own Data](#using-your-own-data)
- [Tips](#tips)
- [Frequently Asked Questions](#frequently-asked-questions)
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

Each configuration contains **6 atoms**: one carbon (C), three hydrogens (H), and two chlorines (Cl). In every frame the atoms appear in the same order:

| Index (0-based) | Element | Role |
|-----------------|---------|------|
| 0 | C | Central carbon |
| 1 | Cl | Chlorine bonded to carbon at the start (leaving group) |
| 2, 3, 4 | H | Methyl hydrogens |
| 5 | Cl | Incoming chlorine (nucleophile) |

At reaction coordinate 0, the C–Cl(1) distance is about 1.8 Å (bonded) and the C–Cl(5) distance is about 5.3 Å (far apart); at reaction coordinate 1 these values are swapped. Along the reaction coordinate, one C–Cl bond breaks while the other forms, passing through the characteristic trigonal-bipyramidal transition state with the methyl group inverting ("umbrella flip").

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

The code requires Python 3 (3.8 or newer recommended) and the following packages:

| Package | Used for |
|---------|----------|
| [PyTorch](https://pytorch.org/) | Building and training the neural network |
| [NumPy](https://numpy.org/) | Array handling |
| [ASE](https://wiki.fysik.dtu.dk/ase/) | Reading and writing XYZ structure files |
| [Matplotlib](https://matplotlib.org/) | Plotting the results |

A GPU is **not** required.

It is good practice to install the packages in a separate environment so they don't interfere with other projects.

**Option 1 – `venv` + `pip`:**

```bash
python3 -m venv vae-env
source vae-env/bin/activate        # On Windows: vae-env\Scripts\activate
pip install torch numpy ase matplotlib
```

**Option 2 – `conda`:**

```bash
conda create -n vae-env python=3.11
conda activate vae-env
conda install -c conda-forge pytorch numpy ase matplotlib
```

To check that everything is installed correctly:

```bash
python -c "import torch, numpy, ase, matplotlib; print('All packages found')"
```

## Usage

Clone the repository and run the script from inside it:

```bash
git clone https://github.com/kuntalg97/Variational-Autoencoder.git
cd Variational-Autoencoder
python generate_vae.py
```

The script must be run from the repository folder, because it looks for `original_configurations.xyz` in the current directory.

While running, the script prints:

- the shapes of the input tensors, and
- the training loss every 100 epochs, e.g. `Epoch [100/10000], Loss: ...`

A typical run looks like this:

```
configs_tensor shape: torch.Size([71, 18])
reaction_coords_tensor shape: torch.Size([71, 1])
input_data shape: torch.Size([71, 19])
Epoch [100/10000], Loss: ...
Epoch [200/10000], Loss: ...
...
Epoch [10000/10000], Loss: ...
Generated 10000 new configurations and saved to ...
```

The shapes tell you that there are 71 training structures, each with 18 coordinates (6 atoms × 3), plus one reaction-coordinate value. The loss should decrease over the first few thousand epochs and then level off. Because training uses random mini-batches, the exact loss values will differ slightly from run to run.

Training runs comfortably on a standard CPU, since the dataset and the network are both small; a full run typically takes a few minutes on a laptop. When training finishes, a plot window opens. Close the window to end the script.

> **Note:** Running the script writes a new `new_configurations_vae.xyz`, replacing the example file included in the repository. Make a copy first if you want to keep the original example.

## Output

After training, the script produces:

1. **`new_configurations_vae.xyz`** – 10,000 newly generated structures ordered along the reaction coordinate from 0 to 1.
2. **A comparison plot** – the carbon atom's x-coordinate versus the reaction coordinate, with the original structures in red and the generated structures in blue. This gives a quick visual check that the generated structures follow the same trend as the training data.

The generated structures can be viewed with any molecular visualization program that reads XYZ files, for example:

- [VMD](https://www.ks.uiuc.edu/Research/vmd/)
- [OVITO](https://www.ovito.org/)
- ASE's built-in viewer: `ase gui new_configurations_vae.xyz`

In VMD or `ase gui`, the file opens as a trajectory, so you can play it frame by frame and watch the reaction proceed from reactants to products.

## Analyzing the Generated Structures

Since both the original and the generated structures are standard XYZ files, they can be loaded and analyzed with a few lines of Python. The example below tracks the two C–Cl distances along the reaction coordinate, which is a natural way to check that the generated pathway makes chemical sense:

```python
import numpy as np
import matplotlib.pyplot as plt
from ase.io import read

original  = read('original_configurations.xyz', index=':')
generated = read('new_configurations_vae.xyz', index=':')

def c_cl_distances(frames):
    """Return the C–Cl(1) and C–Cl(5) distances (in Å) for every frame."""
    d1 = np.array([atoms.get_distance(0, 1) for atoms in frames])
    d2 = np.array([atoms.get_distance(0, 5) for atoms in frames])
    return d1, d2

rc_orig = np.linspace(0, 1, len(original))
rc_gen  = np.linspace(0, 1, len(generated))
d1_orig, d2_orig = c_cl_distances(original)
d1_gen,  d2_gen  = c_cl_distances(generated)

plt.figure(figsize=(8, 5))
plt.scatter(rc_gen, d1_gen, s=2, alpha=0.3, label='Generated C–Cl(1)')
plt.scatter(rc_gen, d2_gen, s=2, alpha=0.3, label='Generated C–Cl(5)')
plt.plot(rc_orig, d1_orig, 'k-', label='Original C–Cl(1)')
plt.plot(rc_orig, d2_orig, 'k--', label='Original C–Cl(5)')
plt.xlabel('Reaction coordinate')
plt.ylabel('Distance (Å)')
plt.legend()
plt.tight_layout()
plt.show()
```

You should see one C–Cl distance grow as the other shrinks, with the two curves crossing near the middle of the pathway (the transition state).

Other things you may want to do with the generated structures:

- **Select a subset** – e.g. every 100th frame: `read('new_configurations_vae.xyz', index='::100')`.
- **Select a region of the pathway** – e.g. frames near the transition state: `read('new_configurations_vae.xyz', index='4500:5500')`.
- **Convert to another format** – ASE can write many formats used by QM and MD codes, e.g. `write('frames.pdb', frames)` or `write('POSCAR', frames[0], format='vasp')`.
- **Use them as starting points** for single-point QM calculations, to build a training set for a machine-learned potential, or to seed enhanced-sampling simulations.

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

Some general guidelines when changing these:

- **More structures, finer pathway** – increase `num_new_configs`.
- **More variety in the generated structures** – increase the KL weight (e.g. `0.1` → `0.5`) or the latent dimension.
- **Structures closer to the training data** – decrease the KL weight (e.g. `0.1` → `0.01`).
- **Faster test runs** – lower `num_epochs` (e.g. to `1000`) while experimenting, then restore it for the final run.

## Using Your Own Data

To apply the code to a different molecular system, replace `original_configurations.xyz` with your own XYZ trajectory. The input and output sizes of the network adapt automatically to the number of atoms.

Step by step:

1. **Prepare a multi-frame XYZ file.** Each frame should follow the standard XYZ layout:
   ```
   <number of atoms>
   <comment line>
   <element>  <x>  <y>  <z>
   <element>  <x>  <y>  <z>
   ...
   ```
   and the frames are simply written one after another in the same file.
2. **Keep the atoms in the same order in every frame.** The network learns each coordinate position by its index, so atom 0 must always be the same atom, atom 1 the same atom, and so on.
3. **Order the frames along the pathway.** The first frame is assigned reaction coordinate 0 and the last frame reaction coordinate 1, with the others evenly spaced in between.
4. **Point the script to your file** by changing the file name in this line of `generate_vae.py`:
   ```python
   original_trajectory = read('original_configurations.xyz', index=':')
   ```
   You can also change the output file name in the `write(...)` line near the end of the script.
5. **Run the script** as described in [Usage](#usage).

Input structures can come from any source that ASE can read (e.g. a nudged elastic band calculation, a constrained geometry scan, or an MD trajectory). If your data is in another format, convert it to XYZ first:

```python
from ase.io import read, write
frames = read('my_pathway.traj', index=':')   # any ASE-readable format
write('original_configurations.xyz', frames)
```

## Tips

### Saving the plot to a file

The script displays the plot on screen with `plt.show()`. To also save it as an image (for example, when running on a remote machine without a display), add this line just before `plt.show()` at the end of `generate_vae.py`:

```python
plt.savefig('vae_comparison.png', dpi=300)
```

On a machine without a display, you can then run the script with a non-interactive Matplotlib backend:

```bash
MPLBACKEND=Agg python generate_vae.py
```

### Reproducible results

Training and generation both involve random numbers, so each run produces slightly different structures. To get identical results every time, add these lines near the top of `generate_vae.py`, right after the imports:

```python
torch.manual_seed(0)
np.random.seed(0)
```

### Saving and reusing the trained model

To avoid retraining every time you want new structures, save the model after training:

```python
torch.save(model.state_dict(), 'vae_model.pt')
```

and later reload it and generate more structures:

```python
model = VAE(input_dim, latent_dim)
model.load_state_dict(torch.load('vae_model.pt'))
model.eval()
with torch.no_grad():
    rcs = torch.linspace(0, 1, 500)          # any reaction-coordinate values in [0, 1]
    z = torch.randn(len(rcs), latent_dim)
    new_configs = model.decode(z, rcs)
```

### Generating structures in a specific region

Because the decoder is conditioned on the reaction coordinate, you can focus generation on any part of the pathway. For example, to sample densely around the transition state, replace the line

```python
new_rcs = torch.linspace(0, 1, num_new_configs)
```

with

```python
new_rcs = torch.linspace(0.4, 0.6, num_new_configs)
```

## Frequently Asked Questions

**Do I need a GPU?**
No. The model and dataset are small, so a standard CPU is enough.

**Why are my generated structures slightly different from the ones in the repository?**
Training and sampling are random. Every run gives a new, equally valid set of structures. See [Reproducible results](#reproducible-results) if you need identical output.

**What units are the coordinates in?**
Ångström (Å), the same as the input XYZ file. The model works directly with the input coordinates, so the output uses whatever units the input uses.

**What does the reaction coordinate represent?**
It is a normalized progress variable from 0 (reactants) to 1 (products), assigned by the order of the frames in the input file.

**Does the model predict energies?**
No. The VAE generates geometries only. Energies and forces for the generated structures can be computed afterwards with a QM code such as CP2K, or with any calculator supported by ASE.

**The script stops after training and doesn't exit.**
It is waiting for the plot window to be closed. Close the window and the script will finish.

## Future Directions

Some natural ways to build on this example:

- **Energy-aware generation** – compute QM energies for the generated structures and use them to filter or weight samples, or add an energy-prediction head to the model.
- **Internal coordinates** – train on bond lengths, angles, and dihedrals (or pairwise distances) instead of Cartesian coordinates, making the model independent of molecular rotation and translation.
- **Larger and more complex systems** – apply the approach to reactions with more atoms, solvent molecules, or multiple reaction pathways.
- **Other generative models** – compare the VAE with normalizing flows, diffusion models, or generative adversarial networks (GANs) for the same task.
- **Training machine-learned potentials** – use the generated structures, labeled with QM energies and forces, as training data for a machine-learned interatomic potential.

## Author

**Kuntal Ghosh**
GitHub: [@kuntalg97](https://github.com/kuntalg97)

Questions and suggestions are welcome — please open an [issue](https://github.com/kuntalg97/Variational-Autoencoder/issues) on GitHub.
