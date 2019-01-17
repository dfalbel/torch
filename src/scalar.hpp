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
