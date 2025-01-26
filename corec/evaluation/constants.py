# ------- #
# METRICS #
# ------- #

CUTOFF_RANX_METRICS = [
    "hits",
    "hit_rate",
    "precision",
    "recall",
    "f1",
    "mrr",
    "map",
    "dcg",
    "dcg_burges",
    "ndcg",
    "ndcg_burges",
]

NON_CUTOFF_RANX_METRICS = [
    "r_precision",
    "bpref",
    "rbp",
]

RANX_METRICS = CUTOFF_RANX_METRICS + NON_CUTOFF_RANX_METRICS

CUSTOM_METRICS = [
    "mean_ctx_sat",
    "acc_ctx_sat",
]

# ---- #
# FUSE #
# ---- #

RANX_FUSE_METHODS = [
    "bayesfuse",
    "bordafuse",
    "anz",
    "gmnz",
    "max",
    "med",
    "min",
    "mnz",
    "sum",
    "condorcet",
    "isr",
    "log_isr",
    "logn_isr",
    "mapfuse",
    "mixed",
    "posfuse",
    "probfuse",
    "rbc",
    "rrf",
    "segfuse",
    "slidefuse",
    "w_bordafuse",
    "w_condorcet",
    "wmnz",
    "wsum",
]

RANX_FUSE_NORMS = [
    "min-max",
    "min-max-inverted",
    "max",
    "sum",
    "zmuv",
    "rank",
    "borda",
]
