# LayerGLAT

This repository is built for our paper _LayerGLAT: a Flexible Non-Autoregressive
Transformer for Single-Pass and Multi-Pass Prediction_

This code is built upon [fairseq](https://github.com/facebookresearch/fairseq), 
you can follow the training instruction of fairseq and set the `--user-dir` to the path of `fairseq_plugin`

The config for training our model can be found under the folder `config`. The define of model class is in the file `fairseq_plugin/models/nat/layer_glat.py`
