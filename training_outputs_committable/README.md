# Committable training outputs

This folder contains selected deployment artifacts that are intentionally kept
outside the ignored `training_outputs/` directory.

- `roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/models/processed_roi8_cnn_only_fold4_seed123.keras`
  - August 13, 2026 fold 4 / seed 123 Keras model.
  - Test accuracy: 0.927038626609442 (92.7%).
- `roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/_e/fold4_s123/test_metrics.json`
  - Metrics supporting the 92.7% result.
- The official training-1-compatible rerun uses the separate ignored output
  namespace `training_outputs/roboflow/mobilenetv3small_8fold_processed_roi_cnn_only_training1_compatible_end_to_end/`.
  It uses the same Roboflow dataset and eight-fold protocol, but its results are
  reported from its own predictions after the RTX 4050 run.
- `roboflow/mobilenetv3small_8samples_final_deployment_cnn_only/models/meatlens_final_8samples_cnn_only_mobilenetv3small.onnx`
  - Existing final-deployment ONNX export.
  - This ONNX file belongs to the separate final deployment run; its recorded test accuracy is 0.9184549356223176 (91.8%), not the fold 4 / seed 123 Keras run above.
