conda create -n meatlens-tf210-gpu python=3.10 -y
conda activate meatlens-tf210-gpu
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 -y
pip install "tensorflow<2.11" jupyterlab ipykernel tf2onnx onnxruntime scikit-learn pandas openpyxl matplotlib seaborn pillow
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"