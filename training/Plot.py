import argparse

from cli import shared_args
from dataclasses import dataclass
from foundations.runner import Runner
import models.registry
from lottery.desc import LotteryDesc
from platforms.platform import get_platform
import pruning.registry
from pruning.mask import Mask
from pruning.pruned_model import PrunedModel
from training import train


class Plot(self):
    def Plot(self):
        model = models.registry.load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs)