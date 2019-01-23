#include "torch_types.h"

template<class T>
torch::Scalar scalar_from_r_impl_ (const SEXP x) {
  auto value = Rcpp::as<T>(x);
  torch::Scalar s_val(value);
  return s_val;
}

torch::Scalar scalar_from_r_ (SEXP x) {
  switch (TYPEOF(x)) {
  case INTSXP:
    return scalar_from_r_impl_<int>(x);
  case REALSXP:
    return scalar_from_r_impl_<double>(x);
  case LGLSXP:
    return scalar_from_r_impl_<bool>(x);
  case CHARSXP:
    Rcpp::stop("strings are not handled yet");
  default:
    Rcpp::stop("not handled");
  }
};

torch::ScalarType scalar_type_from_string(std::string scalar_type) {
  if (scalar_type == "int32" | scalar_type == "int") {
    return torch::kInt;
  } else if (scalar_type == "float64" | scalar_type == "double") {
    return torch::kDouble;
  } else if (scalar_type == "float32" | scalar_type == "float") {
    return torch::kFloat;
  } else if (scalar_type == "uint8") {
    return torch::kByte;
  }
  Rcpp::stop("scalar not handled");
}

std::string scalar_type_to_string(torch::ScalarType scalar_type) {
  if (scalar_type == torch::kInt) {
    return "int";
  } else if (scalar_type == torch::kDouble) {
    return "double";
  } else if (scalar_type == torch::kFloat) {
    return "float";
  }
  Rcpp::stop("scalar not handled");
}
