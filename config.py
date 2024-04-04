class Config:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

# Configurations

GLOBAL_CONFIG = Config(
    seed=1337,
    log_level=1,
    log_file=None
)

RESERVOIR_CONFIG = Config(
    reservoir_type="oscillator",
    nodes=20,
)

DATA_LOADER_CONFIG = Config(
    rows=200,
    test_ratio=0.2,
    encode_labels=True,
    normalize=True,
    repeat_targets=False,
    data_dir='data/ecg'
)

CLASSIFIER_CONFIG = Config(
    ridge=1e-5,
    use_multiprocessing=True,
    save_states=True,
    load_states=True,
)

VISUALIZER_CONFIG = Config(
    results_path="results/",
    dpi=500,
    plot_states=True,
    plot_distribution=False,
)
