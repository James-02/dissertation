# Reservoir Computing with Genetic Oscillators for Arrhythmia Classification

This project focuses on utilizing reservoir computing with genetic oscillators for arrhythmia classification. We provide a range of well-documented modules to aid in this task, from dataset preprocessing, to classification, to metrics visualization.

## Tutorial
Notably, we cover the entire process within a tutorial juypter notebook:
- The `tutorial.ipynb` Jupyter notebook serves as a comprehensive tutorial for using our reservoir computing-based approach for arrhythmia classification, providing robust methodology with visualizations.

## Modules 
- **Reservoir**: Contains the implementation of the genetic oscillator-based reservoir computer, based upon a system of delay differential equations, encapsulated within a `reservoirpy` Node.

- **Utils**: Contains modules for preprocessing, classification, and visualization related to arrhythmia classification using reservoir computing.

- **Optimization**: Includes executable scripts for hyperparameter optimization using Optuna. Additionally, a SLURM script is provided as an example of how to use it with computing clusters.

## Environment Setup

To set up your local environment, follow these steps:

```bash
# Create virtual environment
python -m venv venv

# Install dependencies
venv/bin/pip install -r requirements.txt

# Setup output directories
mkdir -p logs
mkdir -p results/ecg
mkdir -p results/folds
mkdir -p results/metrics
mkdir -p results/preprocessing
mkdir -p results/states
mkdir -p results/training
mkdir -p results/runs
mkdir -p results/forecasting
```

## References

- **ReservoirPy**: This reservoir computing library is instrumental for our reservoir computing implementation, our `OscillatorReservoir` class extends the reservoirpy `Node` class. 

    - [ReservoirPy GitHub](https://github.com/reservoirpy/reservoirpy)

- **Genetic Oscillators**: Our approach builds upon the research into coupled genetic oscillators within colonies of bacteria.
The following papers were instrumental:
  - [A sensing array of radically coupled genetic 'biopixels'](https://doi.org/10.1038/nature10722) by Prindle et al. (2011)
  - [A synchronized quorum of genetic clocks](https://doi.org/10.1038/nature08753) by Danino et al. (2010)
- **Arrhythmia Dataset**:
We take our dataset from Kaggle, which was used within the following paper for arrhythmia classification.
  - [Kaggle Heartbeat Dataset](https://www.kaggle.com/datasets/shayanfazeli/heartbeat) by Shayan Fazeli
  - [ECG Heartbeat Classification: A Deep Transferable Representation](http://dx.doi.org/10.1109/ICHI.2018.00092) by Kachuee et al. (2018)
