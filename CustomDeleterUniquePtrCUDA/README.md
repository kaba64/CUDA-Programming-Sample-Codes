# Using a Custom Deleter for cudaMalloc Allocation

This educational code demonstrates how to use a custom deleter with std::unique_ptr to manage pointers allocated through cudaMalloc.

### Check for Memory Leaks

After compiling the code, you can check for memory leaks by running:

compute-sanitizer  --leak-check=full ./myprogram