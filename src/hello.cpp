#include "torch_types.h"
using namespace Rcpp;

// [[Rcpp::export]]
void create_torch_tensor() {
  torch::Tensor tensor = torch::rand({11, 11, 3});
  auto ten = std::make_shared<torch::Tensor>(tensor);
  Rcout << *ten << std::endl;
};

// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> create_tensor () {

  torch::Tensor tensor = torch::ones({11, 11, 3});
  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
};

// [[Rcpp::export]]
void print_tensor (Rcpp::XPtr<torch::Tensor> x) {
  torch::Tensor b = *x;
  Rcpp::Rcout << b << std::endl;
};

// https://github.com/pytorch/pytorch/issues/14000
// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_from_r (Rcpp::NumericVector x, bool clone = true) {

  auto tensor = torch::from_blob(x.begin(), x.size(), at::kDouble);

  if (clone)
    tensor = tensor.clone();

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
}

