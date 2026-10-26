#include <iostream>

int count_x(const char *p, char x) {
  // Count the number of occurrences of x in p
  if (p == nullptr)
    return 0;
  int count = 0;
  for (; *p != 0; ++p) {
    if (*p == x)
      ++count;
  }
  return count;
}


int main(){
    char p[] = "abcde";
    std::cout << count_x(p,'e') << '\n';
    
    return 0;
}