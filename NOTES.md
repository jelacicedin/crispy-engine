26.10.2026 - Starting out the work


26.10.2026 - Finished Tour of C++ ch. 1 (basics), examples in sandbox/. Next: start ch. 2 (user-defined types). Build+run the open file with Ctrl+Shift+B.


27.10.2026 - In Tour of C++ ch. 3 (modularity). Finished ch. 2 (vector_struct.cpp, vector_class.cpp). Ch. 3 work is in sandbox/modularity/: Vector as a C++20 module (vector.cpp) used via `import Vector; import std;` from user.cpp; builds and links. Toolchain: GCC 16 + CMake + Ninja on Linux (CachyOS), `import std` via libstdc++; was Homebrew LLVM on the Mac; new .cpp files must be added to CMakeLists.txt. Ctrl+Shift+B builds+runs the open file (uses CMake if its folder has a CMakeLists.txt), F5 debugs with breakpoints. Next: continue ch. 3 (separate compilation, namespaces, error handling).
