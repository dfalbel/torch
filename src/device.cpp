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

Rcpp::XPtr<torch::Device> make_device_ptr (torch::Device x) {
  auto * out = new torch::Device(x);
  return Rcpp::XPtr<torch::Device>(out);
}

// Device Index --------

std::int64_t device_index_to_int (torch::DeviceIndex x) {
  return x;
}

torch::DeviceIndex device_index_from_int (std::int64_t x) {
  return x;
}

// Device Type ---------

std::string device_type_to_string (torch::DeviceType x) {
  if (x == torch::DeviceType::CPU) {
    return "CPU";
  } else if (x == torch::DeviceType::CUDA) {
    return "CUDA";
  }
  Rcpp::stop("DeviceType not handled");
}

torch::DeviceType device_type_from_string (std::string x) {
  if (x == "CPU") {
    return torch::DeviceType::CPU;
  } else if (x == "CUDA") {
    return torch::DeviceType::CUDA;
  }
  Rcpp::stop("DeviceType not handled");
}


// Device


// Device attributes

// index

std::int64_t get_device_index (Rcpp::XPtr<torch::Device> device) {
  return device_index_to_int(device->index());
}

// type

std::string get_device_type (Rcpp::XPtr<torch::Device> device) {
  return device_type_to_string(device->type());
}

// Device methods

// has_index

bool device_has_index (Rcpp::XPtr<torch::Device> device) {
  return device->has_index();
}

// is_cuda

bool device_is_cuda (Rcpp::XPtr<torch::Device> device) {
  return device->is_cuda();
}

// is_cpu

bool device_is_cpu (Rcpp::XPtr<torch::Device> device) {
  return device->is_cpu();
}

// ==

bool device_equals (Rcpp::XPtr<torch::Device> device1, Rcpp::XPtr<torch::Device> device2) {
  return (*device1) == (*device2);
}

// set_index

void device_set_index (Rcpp::XPtr<torch::Device> device, std::int64_t index) {
  device->set_index(device_index_from_int(index));
}

