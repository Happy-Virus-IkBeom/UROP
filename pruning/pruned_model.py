# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from models.base import Model
from pruning.mask import Mask

import numpy as np


class PrunedModel(Model):
    @staticmethod
    def to_mask_name(name):
        return 'mask_' + name.replace('.', '___')

    def __init__(self, model: Model, mask: Mask, double_param_level = False, layer_different = False): # optional 변수 적용
        if isinstance(model, PrunedModel): raise ValueError('Cannot nest pruned models.')
        super(PrunedModel, self).__init__()
        self.model = model

        for k in self.model.prunable_layer_names:
            if k not in mask: raise ValueError('Missing mask value {}.'.format(k))
            if not np.array_equal(mask[k].shape, np.array(self.model.state_dict()[k].shape)):
                raise ValueError('Incorrect mask shape {} for tensor {}.'.format(mask[k].shape, k))

        for k in mask:
            if k not in self.model.prunable_layer_names:
                raise ValueError('Key {} found in mask but is not a valid model tensor.'.format(k))

        for k, v in mask.items(): self.register_buffer(PrunedModel.to_mask_name(k), v.float())

        if double_param_level:
            self._apply_mask_double()

        elif layer_different:
            self._apply_layer_different()

        else:
            self._apply_mask()


    def _apply_mask(self):
        for name, param in self.model.named_parameters(): # name: fc_layers.0.weight /fc_layers.1.weight/fc.weight
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))
                #print(type(param.data), '\n', name, '\n', param.data)
                
    ## 내가 추가한것
    def _apply_mask_double(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))
                param.data *= 2 # parameter 을 2배해주기.
    ## 내가 추가한것
    def _apply_layer_different(self):
        for name, param in self.model.named_parameters():
            if hasattr(self, PrunedModel.to_mask_name(name)):
                param.data *= getattr(self, PrunedModel.to_mask_name(name))
                param.data[0] *= 0.5
                param.data[1] *= 2
                param.data[2] *= 3

    def forward(self, x):
        self._apply_mask()
        return self.model.forward(x)

    @property
    def prunable_layer_names(self):
        return self.model.prunable_layer_names

    @property
    def output_layer_names(self):
        return self.model.output_layer_names

    @property
    def loss_criterion(self):
        return self.model.loss_criterion

    def save(self, save_location, save_step):
        self.model.save(save_location, save_step)

    @staticmethod
    def default_hparams(): raise NotImplementedError()
    @staticmethod
    def is_valid_model_name(model_name): raise NotImplementedError()
    @staticmethod
    def get_model_from_name(model_name, outputs, initializer): raise NotImplementedError()
