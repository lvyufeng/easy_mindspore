# Copyright 2024 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""mindnlp parameter"""
from mindspore._c_expression import Tensor as MSTensor
from mindspore.common._stub_tensor import StubTensor
from .._tensor import Tensor

class Parameter(Tensor):
    requires_grad = False
    def __new__(cls, data, requires_grad=True):
        # Ensure data is an instance of Tensor
        if not isinstance(data, (StubTensor, Tensor, MSTensor)):
            raise TypeError("data must be an instance of Tensor")

        # Create a new instance of Parameter
        instance = super(Parameter, cls).__new__(cls)

        if isinstance(data, (Tensor, StubTensor)):
            # Reuse the MSTensor instance from data
            instance.tensor = data.tensor
            instance.stub = data.stub  # Reuse the stub attribute from data
        elif isinstance(data, MSTensor):
            instance.tensor = data
            instance.stub = None

        instance.requires_grad = requires_grad

        return instance

    def __init__(self, data, requires_grad=False): # pylint: disable=super-init-not-called
        # __init__ is called after __new__, can be used for further initialization
        pass
