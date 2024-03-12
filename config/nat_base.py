import logging

from .fairseq_translation import FairseqTranslationTask
from torch.cuda import device_count

logger = logging.getLogger(__name__)


class NATransformerIWSLT14(FairseqTranslationTask):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # tasks
                "--task": "translation_lev",
                "--noise": "full_mask",
                "--eval-bleu-args::iter_decode_max_iter": "1",
                "--eval-bleu-args::iter_decode_with_beam": "1",
                # model
                "--arch": "nonautoregressive_transformer_iwslt_de_en",
                "--dropout": "0.3",
                "--share-all-embeddings": True,
                "--encoder-learned-pos": True,
                "--decoder-learned-pos": True,
                "--pred-length-offset": True,
                "--length-loss-factor": "0.1",
                "--activation-fn": "gelu",
                # criterion
                "--criterion": "nat_loss",
                "--label-smoothing": "0.1",
                # optimizer
                "--optimizer": "adam",
                # lr_scheduler
                "--lr-scheduler": "inverse_sqrt",
                "--warmup-updates": "10000",
                # optimization
                "--max-update": "200000",
                "--clip-norm": "10.0",
                "--lr": "5e-4",
                "--stop-min-lr": "1e-9",
                # checkpoint
                "--no-epoch-checkpoints": True,  # do not save at end_of_epoch
                "--save-interval-updates": "1000",
                "--keep-interval-updates": "10",
                # dataset
                "--train-subset": "train",
                "--max-tokens": "8192",
                "--validate-interval": "0",  # do not validate at end_of_epoch
                "--validate-interval-updates": "1000",
                # NAT configs
                "--apply-bert-init": True,
            }
        )
        return configs

    @property
    def train_debug_configs(self):
        configs = super().train_debug_configs
        configs.update(
            {
                "--eval-bleu-args::decode_with_oracle_length": True,
            }
        )
        return configs

    @property
    def generate(self):
        configs = super().generate
        configs.update(
            {
                # NAT configs
                "--task": "translation_lev",
                "--iter-decode-max-iter": "1",
                "--iter-decode-with-beam": "1",
            }
        )
        return configs


class NATransformerWMT16(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.3",
                # optimization
                "--max-update": "200000",
                "--lr": "5e-4",
                "--update-freq": str(4 // max(1, device_count())),
            }
        )
        return configs


class NATransformerWMT14(NATransformerIWSLT14):
    @property
    def train(self):
        configs = super().train
        configs.update(
            {
                # model
                "--arch": "nonautoregressive_transformer_wmt_en_de",
                "--dropout": "0.1",
                # optimization
                "--max-update": "200000",
                "--lr": "7e-4",
                "--update-freq": str(8 // max(1, device_count())),
            }
        )
        return configs
