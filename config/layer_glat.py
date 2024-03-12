from .nat_base import (
    NATransformerIWSLT14,
    NATransformerWMT14,
    NATransformerWMT16,
)


class NATransformerDEVIWSLT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model specific
                "--arch": "nat_unnamed_iwslt_de_en",
                "--layer-max-iter": "1,",
                "--inference-layer": "-1",
                "--layer-loss-weights": "",
                "--glat-schedule": "0.5,0.5",
                # checkpoint
                "--keep-interval-updates": "10",
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate  # noqa
        configs.update(
            {
                "--model-overrides::arch": "nat_unnamed",
                "--model-overrides::layer_max_iter": "1,",
                "--model-overrides::inference_layer": "-1",
            }
        )
        return configs


class NATransformerDEVWMT14(NATransformerDEVIWSLT14, NATransformerWMT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                "--arch": "nat_unnamed_wmt_en_de",
            }
        )
        return configs


class NATransformerDEVWMT16(NATransformerDEVIWSLT14, NATransformerWMT16):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                "--arch": "nat_unnamed_wmt_en_de",
            }
        )
        return configs
