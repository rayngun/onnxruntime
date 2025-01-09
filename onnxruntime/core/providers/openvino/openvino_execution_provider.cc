// Copyright (C) Intel Corporation
// Licensed under the MIT License
#include <filesystem>
#include <utility>
#include <string>
#include <memory>
#include <vector>
#include <format>
#include "core/providers/shared_library/provider_api.h"
#include "core/providers/openvino/openvino_execution_provider.h"
#include "core/providers/openvino/contexts.h"
#include "core/providers/openvino/backend_manager.h"
#include "core/providers/openvino/onnx_ctx_model_helper.h"
#include "core/providers/openvino/ov_versions/capability.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "openvino/core/version.hpp"
#ifdef USE_OVEP_NPU_MEMORY
#include "core/providers/openvino/ov_allocator.h"
#endif

#define MEMCPY_S(dest, src, destsz, srcsz) memcpy(dest, src, std::min(destsz, srcsz))

namespace onnxruntime {
openvino_ep::SessionContext GetSessionContext(const OpenVINOExecutionProviderInfo& info) {
  openvino_ep::SessionContext result = {
      .enable_opencl_throttling = info.enable_opencl_throttling_,
      .disable_dynamic_shapes = info.disable_dynamic_shapes_,
      .so_context_embed_mode = info.so_context_embed_mode_,
      .so_share_ep_contexts = info.so_share_ep_contexts_,
      .so_context_enable = info.so_context_enable_,
      .enable_qdq_optimizer = info.enable_qdq_optimizer_,
      .so_disable_cpu_ep_fallback = info.so_disable_cpu_ep_fallback_,
      .num_of_threads = info.num_of_threads_,
      .device_type = info.device_type_,
      .precision_str = info.precision_,
      .cache_dir = info.cache_dir_,
      .load_config = info.load_config_,
      .model_priority = info.model_priority_,
      .num_streams = info.num_streams_,
      .context = info.context_,
      .OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR},
      .openvino_sdk_version = std::format("{}.{}", OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR),
  };
  return result;
}

OpenVINOExecutionProvider::OpenVINOExecutionProvider(const OpenVINOExecutionProviderInfo& info)
    : IExecutionProvider{onnxruntime::kOpenVINOExecutionProvider},
      session_context_{GetSessionContext(info)},
      ep_ctx_handle_{session_context_.openvino_sdk_version, *GetLogger()} {
  InitProviderOrtApi();

  // to check if target device is available
  // using ie_core capability GetAvailableDevices to fetch list of devices plugged in
  if (info.cache_dir_.empty()) {
    bool device_found = false;
    std::vector<std::string> available_devices = session_context_.ie_core.GetAvailableDevices();
    // Checking for device_type configuration
    if (info.device_type_ != "") {
      if (info.device_type_.find("HETERO") != std::string::npos ||
          info.device_type_.find("MULTI") != std::string::npos ||
          info.device_type_.find("AUTO") != std::string::npos) {
        device_found = true;
      } else {
        for (const std::string& device : available_devices) {
          if (device.rfind(info.device_type_, 0) == 0) {
            if (info.device_type_.find("GPU") != std::string::npos && (info.precision_ == "FP32" ||
                                                                       info.precision_ == "FP16" ||
                                                                       info.precision_ == "ACCURACY")) {
              device_found = true;
              break;
            }
            if (info.device_type_ == "CPU" && (info.precision_ == "FP32")) {
              device_found = true;
              break;
            }
            if (info.device_type_.find("NPU") != std::string::npos) {
              device_found = true;
              break;
            }
          }
        }
      }
    }
    if (!device_found) {
      ORT_THROW("[ERROR] [OpenVINO] Specified device - " + info.device_type_ + " is not available");
    }
  }
}

std::vector<std::unique_ptr<ComputeCapability>>
OpenVINOExecutionProvider::GetCapability(const GraphViewer& graph_viewer,
                                         const IKernelLookup& /*kernel_lookup*/) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;

  // Enable CI Logs
  if (!(GetEnvironmentVar("ORT_OPENVINO_ENABLE_CI_LOG").empty())) {
    std::cout << "In the OpenVINO EP" << std::endl;
  }

  openvino_ep::GetCapability obj(ep_ctx_handle_,
                                 graph_viewer,
                                 session_context_.device_type,
                                 session_context_.enable_qdq_optimizer);
  result = obj.Execute();

  return result;
}

common::Status OpenVINOExecutionProvider::Compile(
    const std::vector<FusedNodeAndGraph>& fused_nodes,
    std::vector<NodeComputeInfo>& node_compute_funcs) {
  auto& logger = *GetLogger();
  Status status = Status::OK();

  // Assume these properties are constant for all the model subgraphs, otherwise move to SubGraphContext
  session_context_.onnx_model_path_name = fused_nodes[0].filtered_graph.get().ModelPath().string();
  session_context_.onnx_opset_version =
      fused_nodes[0].filtered_graph.get().DomainToVersionMap().at(kOnnxDomain);

  for (const FusedNodeAndGraph& fused_node_graph : fused_nodes) {
    const GraphViewer& graph_body_viewer = fused_node_graph.filtered_graph;
    const Node& fused_node = fused_node_graph.fused_node;

    NodeComputeInfo compute_info;

    session_context_.use_api_2 = true;

    // During backend creation, we check if user wants to use precompiled blob onnx model or the original model
    // For precompiled blob, directly load the model instead of compiling the model
    // For original model, check if the user wants to export a model with pre-compiled blob

    auto& backend_manager = backend_managers_.emplace_back(session_context_,
                                                           fused_node,
                                                           graph_body_viewer,
                                                           logger,
                                                           ep_ctx_handle_);

    compute_info.create_state_func =
        [&backend_manager](ComputeContext* context, FunctionState* state) {
          OpenVINOEPFunctionState* p = new OpenVINOEPFunctionState(backend_manager);
          p->allocate_func = context->allocate_func;
          p->destroy_func = context->release_func;
          p->allocator_handle = context->allocator_handle;
          *state = static_cast<FunctionState>(p);
          return 0;
        };
    compute_info.compute_func = [](FunctionState state, const OrtApi* /* api */, OrtKernelContext* context) {
      auto function_state = static_cast<OpenVINOEPFunctionState*>(state);
      try {
        function_state->backend_manager.Compute(context);
      } catch (const std::exception& ex) {
        return common::Status(common::ONNXRUNTIME, common::FAIL, ex.what());
      }
      return Status::OK();
    };

    compute_info.release_state_func =
        [](FunctionState state) {
          if (state) {
            OpenVINOEPFunctionState* function_state = static_cast<OpenVINOEPFunctionState*>(state);
            delete function_state;
          }
        };
    node_compute_funcs.push_back(compute_info);

    if (!status.IsOK()) {
      break;
    }
  }

  return status;
}

#ifdef USE_OVEP_NPU_MEMORY
std::vector<AllocatorPtr> OpenVINOExecutionProvider::CreatePreferredAllocators() {
  if (session_context_.device_type.find("NPU") != std::string::npos) {
    AllocatorCreationInfo npu_allocator_info{
        [this](OrtDevice::DeviceId device_id) {
          return std::make_unique<OVRTAllocator>(
              session_context_.ie_core.Get(),
              OrtDevice::NPU,
              device_id,
              OpenVINO_RT_NPU);
        },
        0,
    };

    // fill in allocator
    return std::vector<AllocatorPtr>{CreateAllocator(npu_allocator_info)};
  } else {
    return std::vector<AllocatorPtr>{};
  }
}
#endif

common::Status OpenVINOExecutionProvider::SetEpDynamicOptions(gsl::span<const char* const> keys,
                                                              gsl::span<const char* const> values) {
  std::string workload_type = "";
  // Ensure the number of keys and values match
  if (keys.size() != values.size()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT, "Mismatched keys and values sizes.");
  }

  for (size_t i = 0; i < keys.size(); ++i) {
    std::string key = keys[i];
    std::string value = values[i];

    if (key == kOrtEpDynamicOptionsWorkloadType) {
      if (value == "Efficient") {
        workload_type = "EFFICIENT";
      } else if (value == "Default") {
        workload_type = "DEFAULT";
      } else {
        LOGS_DEFAULT(WARNING) << "Unknown workload_type - ignoring " << key << "/" << value;
        LOGS_DEFAULT(WARNING) << "Supported types are 'Efficient' and 'Default' \n";
      }
      if (workload_type != "") {
        LOGS_DEFAULT(INFO) << "SetEpDynamicOptions - modifying: " << key << "/" << value;
        for (auto& backend : backend_managers_) {
          ov::CompiledModel& ov_compiled_model = backend.GetOVCompiledModel();
          ov_compiled_model.set_property(ov::workload_type(workload_type));
        }
      }
    } else {
      // Handle unknown options
      LOGS_DEFAULT(WARNING) << "Unknown key/value pair - ignoring " << key << "/" << value;
    }
  }
  return Status::OK();
}

const InlinedVector<const Node*> OpenVINOExecutionProvider::GetEpContextNodes() const {
  return ep_ctx_handle_.GetEPCtxNodes();
}

}  // namespace onnxruntime
