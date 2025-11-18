#include <iostream>
using namespace std;

// Simple function to add two numbers
int add(int a, int b) {
    return a + b;
}

int main() {
    int x = 10;
    int y = 20;
    int sum = add(x, y);
    
    cout << "Sum: " << sum << endl;
    
    // Conditional statement
    if (sum > 25) {
        cout << "Sum is greater than 25" << endl;
    } else {
        cout << "Sum is less than or equal to 25" << endl;
    }
    
    // Loop example
    for (int i = 0; i < 5; i++) {
        cout << "Iteration: " << i << endl;
    }
    
    return 0;
}