# ----------------------------------------------------------------------------
# Copyright 2014 Nervana Systems Inc.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ----------------------------------------------------------------------------
"""
Sum of squares transform functions and classes.
"""

import numpy as np
from neon.transforms.cost import Cost
from neon.util.param import opt_param


def dqn_loss(backend, outputs, targets, temp, clamping,
             scale_by_batchsize=False):
    """
    Evaluates DQN loss function on pairwise elements from outputs and
    targets.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.
        clamping (float): clamping factor on the errors.

    Returns:
        scalar: Calculated sum of squared diff values for each element.
    """
    actions = backend.zeros((1, targets.shape[1]), dtype=np.int32)
    actions = backend.argmax(targets, axis=0, out=actions)
    error = backend.zeros(outputs.shape)
    place = backend.zeros((1, 1), dtype=np.int32)
    diff = backend.zeros((1, 1))
    for a in range(actions.shape[1]):
        place = actions[0, a]
        diff = backend.subtract(targets[place, a], outputs[place, a], diff)
        diff = clamping if diff > clamping else diff
        diff = -clamping if diff < -clamping else diff
        error[place, a] = backend.multiply(diff, diff, diff)
    temp[0] = error
    if scale_by_batchsize:
        backend.divide(temp[0], temp[0].shape[1], temp[0])
    result = backend.zeros((1, 1), dtype=outputs.dtype)
    backend.sum(temp[0], axes=None, out=result)
    return result


def dqn_derivative(backend, outputs, targets, temp, clamping, scale=1.0):
    """
    Applies the derivative of the  DQN loss function to outputs and targets
    (with respect to the outputs), as specified in the Playing Atari with Deep
    reinforcement Learning paper.

    Arguments:
        backend (Backend): The backend class to use for computation.
        outputs (array_like): predicted output values to be compared.
        targets (array_like): known outcome values to be compared against.
        temp (array_like): temporary buffers.
        clamping (float): clamping factor on the errors.
    """
    actions = backend.zeros((1, targets.shape[1]), dtype=np.int32)
    actions = backend.argmax(targets, axis=0, out=actions)
    error = backend.zeros(outputs.shape)
    place = backend.zeros((1, 1), dtype=np.int32)
    diff = backend.zeros((1, 1))
    for a in range(actions.shape[1]):
        place = actions[0, a]
        diff = backend.subtract(targets[place, a], outputs[place, a], diff)
        diff = clamping if diff > clamping else diff
        diff = -clamping if diff < -clamping else diff
        error[place, a] = diff
    temp[0] = error
    backend.multiply(temp[0], scale, out=temp[0])
    return temp[0]


class DQNCost(Cost):

    """
        Implementation of a DQN Reinforcement learning cost.

        Arguments:
            clamping (float): Allows to clamp the error value
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super(DQNCost, self).__init__(**kwargs)
        opt_param(self, ['clamping'], False)

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf]
        self.outputbuf = databuf

    def get_deltabuf(self):
        return self.temp[0]

    def apply_function(self, targets, scale_by_batchsize=False):
        """
        Apply the DQN cost function to the datasets passed.
        """
        result = dqn_loss(self.backend, self.outputbuf, targets,
                          self.temp, self.clamping, scale_by_batchsize)
        return self.backend.multiply(result, self.scale, result)

    def apply_derivative(self, targets):
        """
        Apply the derivative of the DQN cost function to the datasets passed.
        """
        return dqn_derivative(self.backend, self.outputbuf, targets, self.temp,
                              self.clamping, self.scale)
