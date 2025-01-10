// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <vector>
#include <map>
#include <unordered_map>
#include <string>
#include <filesystem>
#include "core/providers/openvino/ov_interface.h"

namespace onnxruntime {
namespace openvino_ep {

namespace fs = std::filesystem;

struct SharedContext {
  struct shared_weight_key {
    std::string_view name;
    std::string location;
  };
  struct shared_weight_value {
    unsigned int data_offset;
    unsigned int size;
    ov::Tensor* tensor;
  };
  std::map<shared_weight_key, shared_weight_value> shared_weight_map;
  fs::path bin_pathname;
};

using config_t = std::map<std::string, ov::AnyMap>;

struct ProviderInfo {
  std::string device_type{""};             // [device_type]: Overrides the accelerator hardware type and
                                           // precision with these values at runtime.
  std::string precision{""};               // [precision]: Sets the inference precision for execution.
                                           // Supported precision for devices are
                                           // CPU=FP32, GPU=FP32,FP16, NPU=FP16.
                                           // Not setting precision will execute with optimized precision for
                                           // best inference latency. set Precision=ACCURACY for executing
                                           // models with input precision for best accuracy.
  uint32_t num_of_threads{0};              // [num_of_threads]: Overrides the accelerator default value of
                                           // number of threads with this value at runtime.
  config_t load_config{};                  // JSON config map to load custom OV parameters.
  fs::path cache_dir{""};                  // [cache_dir]: specify the path to
                                           // dump and load the blobs for the model caching/kernel caching
                                           // (GPU) feature. If blob files are already present,
                                           // it will be directly loaded.
  std::string model_priority{"DEFAULT"};   // High-level OpenVINO model priority hint
                                           // Defines what model should be provided with more performant
                                           // bounded resource first
  uint32_t num_streams{1};                 // [num_streams]: Option that specifies the number of parallel
                                           // inference requests to be processed on a given `device_type`.
                                           // Overrides the accelerator default value of number of streams
                                           // with this value at runtime.
  void* context{nullptr};                  // OpenCL context
  bool enable_opencl_throttling{false};    // [enable_opencl_throttling]: Enables OpenCL queue throttling for
                                           // GPU device (Reduces CPU Utilization when using GPU)
  bool disable_dynamic_shapes{false};      // [disable_dynamic_shapes]:  Rewrite dynamic shaped models to
                                           // static shape at runtime and execute.
  bool enable_qdq_optimizer{false};        // Enables QDQ pruning for efficient inference latency with NPU
  bool so_context_enable{false};           // ORT session option
  bool so_disable_cpu_ep_fallback{false};  // ORT session option
  bool so_context_embed_mode{false};       // ORT session option
  bool so_share_ep_contexts{false};        // ORT session option
};

// Holds context applicable to the entire EP instance.
struct SessionContext : ProviderInfo {
  SessionContext(const ProviderInfo& info) : ProviderInfo{info} {}

  OVCore ie_core;
  std::vector<bool> deviceAvailableList = {true, true, true, true, true, true, true, true};
  std::string onnx_model_name;
  std::string onnx_model_path_name;
  int onnx_opset_version;
  bool use_api_2;
  const std::vector<int> OpenVINO_Version = {OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR};
  const std::string openvino_sdk_version = std::format("{}.{}", OPENVINO_VERSION_MAJOR, OPENVINO_VERSION_MINOR);
};

// Holds context specific to subgraph.
struct SubGraphContext {
  bool has_dynamic_input_shape = false;
  bool enable_batching = false;
  bool set_npu_config = false;
  bool is_constant = false;
  void* context = 0;
  std::string subgraph_name;
  std::vector<int> input_indexes;
  std::unordered_map<std::string, int> input_names;
  std::unordered_map<std::string, int> output_names;
  bool is_wholly_supported_graph = false;
  bool has_external_weights = false;
  std::string model_precision;
  bool is_ep_ctx_graph = false;
};

}  // namespace openvino_ep
}  // namespace onnxruntime
