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
Cross entropy transform functions and classes.
"""

from neon.transforms.cost import Cost
from neon.util.param import opt_param


class DQNCost(Cost):

    """
    Embodiment of a DQN cost function.

    Warning: This is extremely experimental, and probably that a lot of
    functionalities do not work.

    TODO: Base it on SumSquaredDiff, to ensure compatibility.
    """

    def __init__(self, **kwargs):
        opt_param(self, ['epsilon'], 2 ** -23)
        # default float32 machine epsilon
        super(DQNCost, self).__init__(**kwargs)

    def initialize(self, kwargs):
        opt_param(self, ['shortcut_deriv'], True)
        # raw label indicates whether the reference labels are indexes (raw)
        # or one-hot (default)
        super(DQNCost, self).initialize(kwargs)

    def __str__(self):
        return ("Cost Function: DQNCost")

    def set_outputbuf(self, databuf):
        if not self.outputbuf or self.outputbuf.shape != databuf.shape:
            tempbuf = self.backend.empty(databuf.shape, self.temp_dtype)
            self.temp = [tempbuf]
        self.outputbuf = databuf

    def get_deltabuf(self):
        # used by layer2 only.
        return self.temp[0]

    def apply_function(self, targets, scale_by_batchsize=False):
        pass

    def apply_derivative(self, targets):
        self.temp[0] = self.backend.power(self.temp[0], 0.5, self.temp[0])
        self.temp[0] = self.backend.multiply(self.temp[0], 2.0, self.temp[0])

    def set_cost(self, cost):
        self.temp[0] = cost
