#include "torch_types.h"

torch::Device device_from_string(std::string device) {
  if (device == "CPU") {
    return torch::Device(torch::DeviceType::CPU);
  } else if (device == "CUDA") {
    return torch::Device(torch::DeviceType::CUDA);
  }
  Rcpp::stop("device not handled");
}

std::string device_to_string (torch::Device x) {
  if (x.is_cpu()) {
    return "CPU";
  } else if (x.is_cuda()){
    return "CUDA";
  };
  Rcpp::stop("not handled");
}
