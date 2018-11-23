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
void print_tensor (SEXP a) {
  Rcpp::XPtr<torch::Tensor> x(a);
  torch::Tensor b = *x;
  Rcpp::Rcout << b << std::endl;
};

// https://github.com/pytorch/pytorch/issues/14000
// [[Rcpp::export]]
Rcpp::XPtr<torch::Tensor> tensor_from_r (const Rcpp::NumericVector &x) {

  auto std_x = Rcpp::as<std::vector<double>>(x);

  auto tensor = torch::from_blob(std_x.data(), {5}, at::kDouble);
  tensor = tensor.clone();

  auto * ten = new torch::Tensor(tensor);
  auto ptr = Rcpp::XPtr<torch::Tensor>(ten);

  return ptr;
}

