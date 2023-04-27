// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/openvino/openvino_execution_provider.h"
#include "contrib_ops/openvino/beam_search.h"

namespace onnxruntime {
namespace contrib {
namespace openvino_ep {

ONNX_OPERATOR_KERNEL_EX(
    BeamSearch,
    kMSDomain,
    1,
    kOpenVINOExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::GetTensorType<T>()),
    transformers::BeamSearch BeamSearch);

BeamSearch::BeamSearch(const OpKernelInfo& info)
    : onnxruntime::contrib::transformers::BeamSearch(info) {
 
}

Status BeamSearch::ComputeInternal(OpKernelContext* context) const {
  return onnxruntime::contrib::transformers::BeamSearch::Compute(context);
}

Status BeamSearch::Compute(OpKernelContext* context) const {
  auto s = ComputeInternal(context);
  return s;
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
