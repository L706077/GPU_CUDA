#include <stdlib.h>
#include <string.h>
#include <stdio.h>
extern "C" void kernel_wrapper(int *a, int *b);

int main(int argc, char *argv[]){
    int a = 2;
    int b = 3;

    printf("Input: a = %d, b = %d\n",a,b);
    kernel_wrapper(&a, &b);
    printf("Ran: a = %d, b = %d\n",a,b);
    return 0;
}
