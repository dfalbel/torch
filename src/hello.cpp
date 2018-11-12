#include "torch_types.h"
using namespace Rcpp;

// [[Rcpp::export]]
void create_torch_tensor() {
  torch::Tensor tensor = torch::rand({11, 11, 3});
  auto ten = std::make_shared<torch::Tensor>(tensor);
  Rcout << *ten << std::endl;
};

// [[Rcpp::export]]
SEXP create_tensor () {

  torch::Tensor tensor = torch::rand({11, 11, 3});
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
