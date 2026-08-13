# Roboflow 8-Fold Methodology Amendment

The thesis methodology is authoritative: selecting the Roboflow source must still produce the project’s deterministic 8-fold train/validation/test rotations.

Roboflow’s native `train`, `valid`, and `test` directory names remain in the imported manifest as `roboflow_split` provenance, but they do not define the evaluation folds. Notebook 03 will pool all labeled Roboflow rows and assign them to eight deterministic, stratified partitions. For fold `i`, partition `i` is test, partition `i+1` is validation, and the remaining six partitions are training. This preserves disjoint evaluation sets and mirrors the existing current-dataset rotation shape.

Notebook 04 will select all eight folds by default for Roboflow. Notebook 06 will use the generated fold-1 train/validation files for its final deployment training path rather than reverting to Roboflow’s native split.
