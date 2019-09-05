#include "torch_types.h"
#include "scalar.hpp"

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x) {
  auto * out = new torch::Tensor(x);
  return Rcpp::XPtr<torch::Tensor>(out);
}

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x) {
  std::vector<torch::Tensor> out;
  for (int i = 0; i < x.size(); i++) {
    auto tmp = Rcpp::as<Rcpp::XPtr<torch::Tensor>>(x[i]);
    out.push_back(*tmp);
  }
  return out;
}

std::vector<torch::Tensor> tensor_list_from_r_(Rcpp::Nullable<Rcpp::List> x) {
  if (x.isNull()) {
    return {};
  } else {
    return tensor_list_from_r_(Rcpp::as<Rcpp::List>(x));
  }
}

template<class T>
torch::optional<T> resolve_null_argument (Rcpp::Nullable<T> x) {
  if (x.isNull()) {
    return torch::nullopt;
  } else {
    return Rcpp::as<T>(x);
  }
}

torch::optional<torch::Scalar> resolve_null_scalar (SEXP x) {
  if (Rf_isNull(x)) {
    return torch::nullopt;
  } else {
    return scalar_from_r_(x);
  }
}



