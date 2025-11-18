#include <iostream>
using namespace std;

/* Multi-line comment
   This program demonstrates
   various C++ features */

// Calculate factorial
int factorial(int n) {
    if (n <= 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

// Check if number is even
bool isEven(int num) {
    return num % 2 == 0;
}

int main() {
    int number = 5;
    float pi = 3.14159;
    double e = 2.71828;
    char grade = 'A';
    
    cout << "Factorial of " << number << " is: " << factorial(number) << endl;
    
    // While loop
    int count = 0;
    while (count < 3) {
        cout << "Count: " << count << endl;
        count++;
    }
    
    // Array example
    int arr[5] = {1, 2, 3, 4, 5};
    
    for (int i = 0; i < 5; i++) {
        if (isEven(arr[i])) {
            cout << arr[i] << " is even" << endl;
        } else {
            cout << arr[i] << " is odd" << endl;
        }
    }
    
    // Logical operators
    bool flag1 = true;
    bool flag2 = false;
    
    if (flag1 && !flag2) {
        cout << "Both conditions satisfied" << endl;
    }
    
    return 0;
}