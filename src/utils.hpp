#include "torch_types.h"

Rcpp::XPtr<torch::Tensor> make_tensor_ptr (torch::Tensor x);

Rcpp::List tensorlist_to_r (torch::TensorList x);

Rcpp::XPtr<torch::QScheme> make_qscheme_ptr (torch::QScheme x);

std::vector<torch::Tensor> tensor_list_from_r_ (Rcpp::List x);

std::vector<torch::Tensor> tensor_list_from_r_(Rcpp::Nullable<Rcpp::List> x);

torch::optional<std::int64_t> resolve_null_argument (Rcpp::Nullable<std::int64_t> x);

torch::optional<torch::Scalar> resolve_null_scalar (SEXP x);





