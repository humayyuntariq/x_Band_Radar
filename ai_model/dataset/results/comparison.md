## 1. Model comparison (all features)
| model | cv_F1 | test_acc | test_F1 | test_adj | test_MAE | nat_acc | nat_F1 | edge_F1 |
|---|---|---|---|---|---|---|---|---|
| gbm | 0.882 | 0.881 | 0.881 | 0.963 | 0.165 | 0.925 | 0.772 | 0.705 |
| rf | 0.875 | 0.877 | 0.877 | 0.955 | 0.185 | 0.898 | 0.709 | 0.674 |
| mlp | 0.859 | 0.860 | 0.859 | 0.959 | 0.189 | 0.908 | 0.757 | 0.641 |
| coarsest_feasible |  | 0.708 | 0.699 | 0.875 | 0.510 | 0.489 | 0.330 | 0.368 |
| range_scale_rule |  | 0.222 | 0.154 | 0.567 | 1.394 | 0.019 | 0.032 | 0.009 |
| always_mode1 |  | 0.201 | 0.067 | 0.398 | 1.998 | 0.827 | 0.181 | 0.178 |

## 2. Constraint safety and cost (natural mix)
| model | viol_raw | viol_after | warn_infeas | ms_per_1k | model_KB | fit_s |
|---|---|---|---|---|---|---|
| gbm | 0.0038 | 0.0012 | 0.9275 | 13.0078 | 1093.5820 | 6.5548 |
| rf | 0.0027 | 0.0014 | 0.9275 | 6.6817 | 16076.9160 | 1.4135 |
| mlp | 0.0076 | 0.0011 | 0.9275 | 0.9791 | 290.1777 | 5.5021 |
| coarsest_feasible | 0.0081 | 0.0079 | 0.9275 | 0.1162 | 0.0000 |  |
| range_scale_rule | 0.6900 | 0.0061 | 0.9275 | 0.0752 | 0.0000 |  |
| always_mode1 | 0.0464 | 0.0000 | 0.9275 | 0.0522 | 0.0000 |  |

## 3. Ablation: which inputs does the classifier need? (macro-F1)
| model | edge_F1:env_only | edge_F1:full | edge_F1:nav_only | edge_F1:raw | test_F1:env_only | test_F1:full | test_F1:nav_only | test_F1:raw |
|---|---|---|---|---|---|---|---|---|
| gbm | 0.309 | 0.705 | 0.348 | 0.674 | 0.304 | 0.881 | 0.775 | 0.878 |
| mlp | 0.315 | 0.641 | 0.337 | 0.608 | 0.299 | 0.859 | 0.775 | 0.851 |
| rf | 0.298 | 0.674 | 0.350 | 0.665 | 0.306 | 0.877 | 0.778 | 0.877 |

feature sets: full = env + nav + engineered, raw = env + nav, env_only = environment only, nav_only = range/resolution/RCS only