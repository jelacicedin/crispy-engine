26.10.2026 - Starting out the work


26.10.2026 - Finished Tour of C++ ch. 1 (basics), examples in sandbox/. Next: start ch. 2 (user-defined types). Build+run the open file with Ctrl+Shift+B.


27.10.2026 - In Tour of C++ ch. 3 (modularity). Finished ch. 2 (vector_struct.cpp, vector_class.cpp). Ch. 3 work is in sandbox/modularity/: Vector as a C++20 module (vector.cpp) used via `import Vector; import std;` from user.cpp; builds and links. Toolchain: GCC 16 + CMake + Ninja on Linux (CachyOS), `import std` via libstdc++; was Homebrew LLVM on the Mac; new .cpp files must be added to CMakeLists.txt. Ctrl+Shift+B builds+runs the open file (uses CMake if its folder has a CMakeLists.txt), F5 debugs with breakpoints. Next: continue ch. 3 (separate compilation, namespaces, error handling).

30.10.2026 - In Tour of C++ ch. 6 (essential operations), at 6.2 (copy and move). This is the chapter that makes the Tensor tractable: constructors/destructors and RAII, then value semantics, resource handling and operator overloading. Plan is to start serious engine work once ch. 6 is done.
sandbox/demo_autograd.cpp is a throwaway on the side -- 213 lines of reverse-mode autodiff (op set is just + - * / exp, tanh composed from exp), gradient-checked against central differences and trained on XOR -- written to answer "how hard would gradients be?", not part of the project.
Next: finish ch. 6, then the engine skeleton (CMake + GoogleTest via FetchContent, -fsanitize=address,undefined debug config), then the Tensor (shape, strides, offset, shared_ptr<Storage>, Rule of Zero).
