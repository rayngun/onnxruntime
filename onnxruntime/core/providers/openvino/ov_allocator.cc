// Copyright (C) Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/ov_allocator.h"
#include "core/providers/openvino/ov_interface.h"
#include "openvino/runtime/intel_npu/level_zero/level_zero.hpp"
#include "openvino/runtime/intel_npu/properties.hpp"

namespace onnxruntime {

using namespace openvino_ep;

OVRTAllocator::OVRTAllocator(OrtDevice::DeviceType device_type, OrtDevice::DeviceId device_id, const char* name) : IAllocator(OrtMemoryInfo(name, OrtAllocatorType::OrtDeviceAllocator, OrtDevice(device_type, OrtDevice::MemType::DEFAULT, device_id), device_id, OrtMemTypeCPUInput)) {
    if(device_type == OrtDevice::NPU) {
        ov::Core core;
        remote_ctx_ = core.get_default_context("NPU").as<ov::intel_npu::level_zero::ZeroContext>();
    } else {
        ORT_THROW("Invalid device type");
    }
}

void* OVRTAllocator::Alloc(size_t size) {
  try {
      // TODO: probably want to handle alignment
    ov::Tensor* tensor = new ov::Tensor(remote_ctx_.create_host_tensor(ov::element::Type_t::u8, {size + sizeof(ov::Tensor*)}));
    ov::Tensor** ptr = reinterpret_cast<ov::Tensor**>(tensor->data());
    *ptr = tensor;
    return reinterpret_cast <void*>(ptr + 1);
  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Alloc failed: ") + e.what());
  }
  return nullptr;
}

void OVRTAllocator::Free(void* p) {
  try {
    ov::Tensor **ptr = reinterpret_cast<ov::Tensor**>(p);
    delete ptr[-1];
  } catch (const ov::Exception& e) {
    ORT_THROW(std::string("Free failed: ") + e.what());
    }
  }

}  // namespace onnxruntime
