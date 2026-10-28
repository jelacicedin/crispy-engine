import Vector;
#include <cmath>
#include <iostream>

double sqrt_sum(Vector &v) {
  double sum = 0;
  for (int i = 0; i < v.size(); i++) {
    sum += std::sqrt(v[i]);
  }
  return sum;
}

int main() {
  Vector v(5);
  for (int i = 0; i != 5; i++) {
    std::cin >> v[i];
  }
  std::cout << sqrt_sum(v) << std::endl;
}