# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
#내가 추가한 import 코드
import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
#tensorboard
#from torch.utils.tensorboard import SummaryWriter
#from tensorboardx import SummaryWriter

@dataclass
class LotteryRunner(Runner):
    replicate: int
    levels: int
    desc: LotteryDesc
    verbose: bool = True
    evaluate_every_epoch: bool = True

    @staticmethod
    def description():
        return 'Run a lottery ticket hypothesis experiment.'

    @staticmethod
    def _add_levels_argument(parser):
        help_text = \
            'The number of levels of iterative pruning to perform. At each level, the network is trained to ' \
            'completion, pruned, and rewound, preparing it for the next lottery ticket iteration. The full network ' \
            'is trained at level 0, and level 1 is the first level at which pruning occurs. Set this argument to 0 ' \
            'to just train the full network or to N to prune the network N times.'
        parser.add_argument('--levels', required=True, type=int, help=help_text)

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        # Get preliminary information.
        defaults = shared_args.maybe_get_default_hparams()

        # Add the job arguments.
        shared_args.JobArgs.add_args(parser)
        lottery_parser = parser.add_argument_group(
            'Lottery Ticket Hyperparameters', 'Hyperparameters that control the lottery ticket process.')
        LotteryRunner._add_levels_argument(lottery_parser)
        LotteryDesc.add_args(parser, defaults)

    @staticmethod
    def create_from_args(args: argparse.Namespace) -> 'LotteryRunner':
        return LotteryRunner(args.replicate, args.levels, LotteryDesc.create_from_args(args),
                             not args.quiet, not args.evaluate_only_at_end)

    def display_output_location(self):
        print(self.desc.run_path(self.replicate, 0))

    def run(self) -> None:
        if self.verbose and get_platform().is_primary_process:
            print('='*82 + f'\nLottery Ticket Experiment (Replicate {self.replicate})\n' + '-'*82)
            print(self.desc.display)
            print(f'Output Location: {self.desc.run_path(self.replicate, 0)}' + '\n' + '='*82 + '\n')

        if get_platform().is_primary_process: self.desc.save(self.desc.run_path(self.replicate, 0))
        if self.desc.pretrain_training_hparams: self._pretrain() # pretrain_training_hparams 있으면 pretraining
        if get_platform().is_primary_process: self._establish_initial_weights() # 초기단계면 initial_weight 만들기.
        get_platform().barrier()

        for level in range(self.levels+1):
            if get_platform().is_primary_process: self._prune_level(level) # 프루닝 전단계 파라미터 불러오기
            get_platform().barrier()
            self._train_level(level)  # 프루닝 후 트레이닝 단계


    # Helper methods for running the lottery.
    def _pretrain(self):
        location = self.desc.run_path(self.replicate, 'pretrain')
        if models.registry.exists(location, self.desc.pretrain_end_step): return

        if self.verbose and get_platform().is_primary_process: print('-'*82 + '\nPretraining\n' + '-'*82)
        model = models.registry.get(self.desc.model_hparams, outputs=self.desc.pretrain_outputs)
        train.standard_train(model, location, self.desc.pretrain_dataset_hparams, self.desc.pretrain_training_hparams,
                             verbose=self.verbose, evaluate_every_epoch=self.evaluate_every_epoch)

    def _establish_initial_weights(self):
        location = self.desc.run_path(self.replicate, 0)
        if models.registry.exists(location, self.desc.train_start_step): return

        new_model = models.registry.get(self.desc.model_hparams, outputs=self.desc.train_outputs)

        # If there was a pretrained model, retrieve its final weights and adapt them for training.
        if self.desc.pretrain_training_hparams is not None:
            pretrain_loc = self.desc.run_path(self.replicate, 'pretrain')
            old = models.registry.load(pretrain_loc, self.desc.pretrain_end_step,
                                       self.desc.model_hparams, self.desc.pretrain_outputs)
            state_dict = {k: v for k, v in old.state_dict().items()}

            # Select a new output layer if number of classes differs.
            if self.desc.train_outputs != self.desc.pretrain_outputs:
                state_dict.update({k: new_model.state_dict()[k] for k in new_model.output_layer_names})

            new_model.load_state_dict(state_dict)

        new_model.save(location, self.desc.train_start_step) # new_model, train_start_step 저장.

    def _train_level(self, level: int):
        location = self.desc.run_path(self.replicate, level) # 해당 level path

        if models.registry.exists(location, self.desc.train_end_step):

            # image PATH 가 없으면 make directory => 내가 추가한것.
            if not os.path.exists(f'{location}\Distribution_of_Weight'):
                os.makedirs(f'{location}\Distribution_of_Weight')

                IMAGE_PATH = f'{location}\Distribution_of_Weight'
                # Weight Load & Weight Plotting => 내가 추가

                print('\nPlotting Location is: ',location)
                model = models.registry.load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                             self.desc.model_hparams, self.desc.train_outputs)

                # Load Original_Save Parameter : As the batch size changes, the ep should be adjusted. default: batch_size=16
                for ep,iteration in [[0,0],[149,234]]:
                    model.load_state_dict(torch.load('{}\model_ep{}_it{}.pth'.format(location, ep,iteration)))
                    model.eval()
                    print("\nmodel_ep{}_it{}.pth".format(ep,iteration))


                    for param_tensor in model.state_dict():
                        #print(param_tensor, "\n", model.state_dict()[param_tensor])
                        tensor = model.state_dict()[param_tensor]
                        tensor = tensor.numpy()

                        #tensor에서 weight 만 추출
                        tensor = tensor[0]
                        #print(tensor)
                        tensor = tensor.reshape(-1)
                        sns.kdeplot(tensor)

                        plt.savefig('{}\Distribution_of_weights_level{}_ep{}.png'.format(IMAGE_PATH, level, ep))
                    plt.clf()

            """
            # Load Weights before & after Training => 내가 추가
            for ep in range(14):
                for Label in ["Before","After"]:
                    print("\n {} Training \n".format(Label))
                    model.load_state_dict(torch.load('{}\weights\Record_Weights_{}_ep{}.pth'.format(location,Label, ep)),strict = False)
                    model.eval()

                    for param_tensor in model.state_dict():
                        print(param_tensor, "\t", model.state_dict()[param_tensor])
                        #print(param_tensor, "\t", model.state_dict()[param_tensor].size())
            """

            return

        # 만약 트레이닝을 전에 시켰다면 위에서 return 해서 끝나버려서 여기까지 안옴.
        model = models.registry.load(self.desc.run_path(self.replicate, 0), self.desc.train_start_step,
                                     self.desc.model_hparams, self.desc.train_outputs)
        # level 7일때부터(즉 21 % 남았을때 부터)는 double_param_level = True 이면 초기화시 masking 후 2배 적용,
        # layer_differnt = True 이면 layer 별로 masking 한후 각기 다르게 상수배 해줌.
        if level >= 7 :
            pruned_model = PrunedModel(model, Mask.load(location), double_param_level = False, layer_different = True)
        else:
            pruned_model = PrunedModel(model, Mask.load(location)) # model, mask 불러오기

        pruned_model.save(location, self.desc.train_start_step) # pruned된 모델 저장
        #print(f'Prunded Model is: {PrunedModel(model,Mask.load(location))}\n')
        #print(f'Mask.load is: {Mask.load(location)}')
        if self.verbose and get_platform().is_primary_process:
            print('-'*82 + '\nPruning Level {}\n'.format(level) + '-'*82)   # level = 0, level = 1... 등 pruning level 표시
        """
        # 내가 추가 (tensor 다루는법이 있어서 남겨둠)
        if level > 7:
            pruned_model.eval()
            for param_tensor in pruned_model.state_dict():
                tensor = pruned_model.state_dict()[param_tensor]
                tensor = tensor.numpy()
                tensor = tensor * 2
                pruned_model.state_dict()[param_tensor] = tensor
            pruned_model.save(location, self.desc.train_start_step)
        """

        train.standard_train(pruned_model, location, self.desc.dataset_hparams, self.desc.training_hparams,
                             start_step=self.desc.train_start_step, verbose=self.verbose,
                             evaluate_every_epoch=self.evaluate_every_epoch) # training 하기 => trainig 할때는 trian_start_step의 parameter 사용

    def _prune_level(self, level: int):
        new_location = self.desc.run_path(self.replicate, level)
        if Mask.exists(new_location): return

        if level == 0:
            Mask.ones_like(models.registry.get(self.desc.model_hparams)).save(new_location) # level=0일때는 mask 다 1 => weight다 살리기
        else:
            old_location = self.desc.run_path(self.replicate, level-1) # 아니라면 old location = 직전 level 에 저장된 path인 run_path
            model = models.registry.load(old_location, self.desc.train_end_step,
                                         self.desc.model_hparams, self.desc.train_outputs) # pruning이기때문에 train_end_step 불러오는것임!

            pruning.registry.get(self.desc.pruning_hparams)(model, Mask.load(old_location)).save(new_location) # registry.get 에는 return partial 부분에 .prune이 있어 프루닝이 되고 이후 new_location에 저장.

