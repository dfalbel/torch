#include "torch_types.h"

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x);

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x);

std::vector<torch::Tensor> tensor_list_from_r_(Rcpp::Nullable<Rcpp::List> x);

template<class T>
torch::optional<T> resolve_null_argument (Rcpp::Nullable<T> x);

torch::optional<torch::Scalar> resolve_null_scalar (SEXP x);





