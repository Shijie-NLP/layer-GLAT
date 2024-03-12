import logging
import os


from .fairseq_task import FairseqTask

logger = logging.getLogger(__name__)


class FairseqTranslationTask(FairseqTask):
    """Base Config for all Fairseq Translation Tasks"""

    def __init__(self, model, task):
        # task_name should in the form of `dataset_src_tgt`, e.g. iwslt14_de_en.
        task, self.src, self.tgt = task.strip().split("_")
        self.lang_pair = self.src + "-" + self.tgt

        datasets = DATASETS_REGISTRY[task][self.lang_pair]

        super().__init__(model, task, datasets)

    @property
    def save_dir(self):
        return os.path.join(EXPERIMENT_DIR, self.task, self.lang_pair, self.model)

    @property
    def train(self):
        configs = {
            # tasks
            "--data": "fairseq",
            "--task": "translation",
            "--source-lang": self.src,
            "--target-lang": self.tgt,
            "--max-source-positions": "512",
            "--max-target-positions": "512",
            "--eval-bleu": True,
            "--eval-bleu-print-samples": True,
            # checkpoint
            "--save-dir": self.save_dir,
            "--keep-best-checkpoints": "5",
            "--best-checkpoint-metric": "bleu",
            "--maximize-best-checkpoint-metric": True,
            "--patience": "-1",
            # common
            "--log-interval": "100",
            "--log-format": "simple",
            "--log-file": "log.txt",
            "--tensorboard-logdir": "tensorboard",
            "--fp16": True,
            "--seed": "42",
            # dataset
            "--fixed-validation-seed": "7",
            # distributed
            "--ddp-backend": "legacy_ddp",
        }
        return configs

    def register_train_configs(self, config_parser, dataset):
        # if specified, use specified configs,
        # otherwise use dataset default configs
        tokenizer = CR.pop("--eval-bleu-detok", dataset.TOKENIZER)
        bpe = CR.pop("--eval-bleu-remove-bpe", dataset.BPE)
        CR.update({"--eval-bleu-detok": tokenizer, "--eval-bleu-remove-bpe": bpe})

        configs = config_parser.items(f"--tokenizer.{tokenizer}")
        configs = CR.merge_with_parent(configs, update=False)

        CR.update(
            {"--eval-bleu-detok-args::" + k[2:].replace("-", "_"): v for k, v in configs.items()},
            verbose=False,
        )
        super().register_train_configs(config_parser, dataset)

        return {"--eval-bleu-args"}

    @property
    def generate(self):
        configs = {
            # tasks
            "--data": "fairseq",
            "--task": "translation",
            "--source-lang": self.src,
            "--target-lang": self.tgt,
            "--max-source-positions": "512",
            "--max-target-positions": "512",
            # dataset
            "--batch-size": "128",
            "--required-batch-size-multiple": "1",
            # common
            "--fp16": True,
            # common_eval
            "--path": "checkpoint_best.pt",
            "--results-path": self.save_dir,
            # scoring
            "--scoring": "sacrebleu",
        }
        return configs

    def register_eval_configs(self, config_parser, dataset):
        tokenizer = CR.pop("--tokenizer", dataset.TOKENIZER)
        bpe = CR.pop("--post-process", dataset.BPE)
        CR.update({"--tokenizer": tokenizer, "--post-process": bpe})

        CR.merge_with_parent(config_parser.items(f"--tokenizer.{tokenizer}"))

        super().register_eval_configs(config_parser, dataset)

        return set()
