# DDI Prediction using Knowledge Graph Convolutional Networks

A PyTorch implementation of Knowledge Graph Convolutional Networks (KGCN) for Drug-Drug Interaction (DDI) prediction.

## Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Install Dependencies

1. Clone the repository:
```bash
git clone <repository-url>
cd DDI
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

Or install packages individually:
```bash
pip install torch numpy pandas scikit-learn matplotlib
```

## Overview

This project predicts potential drug-drug interactions using a knowledge graph approach that incorporates:
- Drug similarity features
- Target similarity features  
- Enzyme similarity features
- Knowledge graph relationships between drugs and proteins

## Key Components

- **`KGCN.py`** - Main training script with dataset preparation and model training
- **`model.py`** - KGCN neural network implementation
- **`data_loader.py`** - Data preprocessing and knowledge graph construction
- **`aggregator.py`** - Neighbor aggregation strategies (sum, concat, neighbor)

## Data

The `data/` directory should contain:
- Drug-drug interaction datasets
- Drug, target, and enzyme similarity matrices
- Knowledge graph data linking drugs to proteins
- Drug indices and mappings

**Note**: Due to file size limitations, data files are not included in the repository. Please ensure you have the required data files in the `data/` directory before running the model.

## Usage

```bash
python KGCN.py --n_epochs 3 --batch_size 16 --dim 16 --lr 5e-4
```

## Model Architecture

The KGCN model uses graph convolutional networks to learn drug representations by aggregating information from neighboring entities in the knowledge graph, combined with similarity features for comprehensive DDI prediction.