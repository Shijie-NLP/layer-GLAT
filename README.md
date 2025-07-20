# LayerGLAT

This repository accompanies our ECML 2024 paper:
_LayerGLAT: A Flexible Non-autoregressive Transformer for Single-Pass and Multi-Pass Prediction_

This implementation is based on [fairseq](https://github.com/facebookresearch/fairseq).

## 🧠 Model Code
The model class is defined in: `fairseq_plugin/models/nat/layer_glat.py`

## 🚀 Training
To train the model, follow the standard fairseq training instructions and set the `--user-dir` argument to the path of the `fairseq_plugin` directory in this project.

The training configuration for our model can be found in the `config/` folder. Specifically, you can use the configuration file located at `config/layer_glat.py`


