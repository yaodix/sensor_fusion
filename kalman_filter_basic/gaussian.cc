#include <iostream>
#include <vector>

struct GaussianParams {
  double mean;
  double var;
};

double gaus(float mu, float sigma2, float x) {

}

// 本质：两个高斯函数相乘
GaussianParams Update(double mean1, double var1, double mean2, double var2) {
  return {(mean1*var2 + var1*mean2) / (var1 + var2), (1. / (1./var1 + 1./var2))};
}

// 本质：高斯函数移动
GaussianParams Predict(double mean1, double var1, double mean2, double var2) {
  return  {mean1 + mean2, var1 + var2};
}


int main() {
std::vector<double> measurements{5, 6, 7, 9, 10};
std::vector<double> motion{1, 1, 2, 1, 1};
double measurment_sig = 4;
double motion_sig = 3;

double mu = 0;      // 初始估计值
double sig = 1000;  // 初始估计方差






  return 0;
}