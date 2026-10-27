#include <iostream>
struct Vector {
  double *elem;
  int size;
};

void vector_init(Vector &v, int size) {
  v.elem = new double[size];
  v.size = size;
}

double read_and_sum(int s) {
  Vector v;
  vector_init(v, s);

  for (int i = 0; i < s; i++)
    std::cin >> v.elem[i];

  double sum = 0.0;
  for (int i = 0; i < s; i++) {
    sum += v.elem[i];
  }
  return sum;
}

int main() {
    std::cout << read_and_sum(5) << '\n';
}