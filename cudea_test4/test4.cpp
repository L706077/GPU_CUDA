#include <stdio.h>


//索引用到的緒構體
struct Index{
        int block, thread;
};

extern "C" void kernel_wrapper(Index *aa, Index *bb);


//主函式
int main(){
        Index* aa;
        Index  bb[100];
	kernel_wrapper(aa,bb);

	int g=3, b=4, m=g*b;
        for(int i=0; i<m; i++){
            printf("bb[%d]={block:%d, thread:%d}\n", i,bb[i].block, bb[i].thread);
        }


        return 0;
}
