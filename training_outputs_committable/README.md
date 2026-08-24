# Committable training outputs

This folder contains selected deployment artifacts that are intentionally kept
outside the ignored `training_outputs/` directory.

- `roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/models/processed_roi8_cnn_only_fold4_seed123.keras`
  - August 13, 2026 fold 4 / seed 123 Keras model.
  - Test accuracy: 0.927038626609442 (92.7%).
- `roboflow/mobilenetv3small_8fold_processed_roi_cnn_only/_e/fold4_s123/test_metrics.json`
  - Metrics supporting the 92.7% result.
- `roboflow/mobilenetv3small_8samples_final_deployment_cnn_only/models/meatlens_final_8samples_cnn_only_mobilenetv3small.onnx`
  - Existing final-deployment ONNX export.
  - This ONNX file belongs to the separate final deployment run; its recorded test accuracy is 0.9184549356223176 (91.8%), not the fold 4 / seed 123 Keras run above.
