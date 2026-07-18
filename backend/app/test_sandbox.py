import asyncio
from sandbox import execute_local_code

def run_tests():
    print("--- 🧪 TESTING PYTHON SANDBOX ---")
    py_code = """
import math
print("Python is alive.")
print(f"The square root of 144 is {math.sqrt(144)}")
    """
    py_result = execute_local_code("python", py_code)
    print(py_result)


    print("\n--- 🧪 TESTING C++ SANDBOX ---")
    cpp_code = """
#include <iostream>
using namespace std;

int main() {
    cout << "C++ is alive and compiled successfully!" << endl;
    
    // Quick loop test
    for(int i=1; i<=3; i++) {
        cout << "Iteration: " << i << endl;
    }
    return 0;
}
    """
    cpp_result = execute_local_code("cpp", cpp_code)
    print(cpp_result)


    print("\n--- 🧪 TESTING TIMEOUT/INFINITE LOOP SAFETY ---")
    infinite_loop_code = """
while True:
    pass
    """
    # We set a short 2-second timeout for the test
    timeout_result = execute_local_code("python", infinite_loop_code, timeout=2.0)
    print(timeout_result)

if __name__ == "__main__":
    run_tests()
