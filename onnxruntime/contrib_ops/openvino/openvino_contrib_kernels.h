// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/status.h"

namespace onnxruntime {
   class KernelRegistry;
namespace contrib {
namespace openvino_ep {
Status RegisterOpenVINOContribKernels(KernelRegistry& kernel_registry);
} // namespace OpenVINO
} // namespace contrib
}  // namespace onnxruntime
