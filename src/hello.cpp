#include <torch/torch.h>
#include <Rcpp.h>
using namespace Rcpp;

// [[Rcpp::export]]
IntegerVector hello2() {
  at::Tensor tensor = torch::rand({2, 3});
  Rcout << tensor << std::endl;
  return 1;

};

