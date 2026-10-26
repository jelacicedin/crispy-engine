#include <cstdio>
#include <iostream>

int main() {
    int v1[10] = {0,1,2,3,4,5,6,7,8,9};
    for (auto i=0; i<10; i++){
        std::cout << v1[i] <<'\n';
    }
    std::printf("potato\n");
    for (auto x:v1) {
        std::cout << x << '\n';
    }
}