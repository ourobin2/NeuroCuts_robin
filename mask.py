import tensorflow.compat.v1 as tf
from collections import OrderedDict
import contextlib
import gym
from gym.spaces import Space
import numpy as np
from typing import Dict, List, Any, Union

from ray.rllib.models import ModelV2
from ray.rllib.models.tf.misc import normc_initializer
from ray.rllib.utils.typing import ModelConfigDict, ModelInputDict, TensorStructType


class PartitionMaskModel(ModelV2):
    """ Model that only allows the partitioning action at the first level. """

    def __init__(self,
                 obs_space: Space,
                 action_space: Space,
                 num_outputs: int,
                 model_config: ModelConfigDict,
                 name: str):

        super(PartitionMaskModel, self).__init__(obs_space, action_space, num_outputs,
                                                 model_config, name, framework="tf")

    def _build_layers_v2(self, input_dict, num_outputs, options):
        mask = input_dict["obs"]["action_mask"]

        last_layer = input_dict["obs"]["real_obs"]
        hiddens = options["fcnet_hiddens"]

        for i, size in enumerate(hiddens):
            label = "fc{}".format(i)
            last_layer = tf.layers.dense(
                last_layer,
                size,
                kernel_initializer=normc_initializer(1.0),
                activation=tf.nn.tanh,
                name=label)

        action_logits = tf.layers.dense(
            last_layer,
            num_outputs,
            kernel_initializer=normc_initializer(0.01),
            activation=None,
            name="fc_out")

        if num_outputs == 1:
            return action_logits, last_layer

        # Mask out invalid actions (use tf.float32.min for stability)
        inf_mask = tf.maximum(tf.log(mask), tf.float32.min)
        masked_logits = inf_mask + action_logits

        return masked_logits, last_layer
