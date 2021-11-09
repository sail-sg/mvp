/* Copyright 2021 Garena Online Private Limited.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

/*!
# --------------------------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0 and Deformable DETR
# --------------------------------------------------------------------------------------------------------------------
*/

#include "deform.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("deform_forward", &deform_forward, "deform_forward");
  m.def("deform_backward", &deform_backward, "deform_backward");
}
