YAML_TEMPLATE = {
    "data_config": {
        "strategy": "fixed",
    },
    "dataset": "corec_dataset",
    # Elliot requires an evaluation
    "evaluation": {
        "cutoffs": [5],
        "simple_metrics": ["nDCG"],
    },
    "models": {},
}

META_TEMPLATE = {
    "meta": {
        "save_recs": True,
        "verbose": False,
    }
}
