# GPU_CUDA
***CUDA=Compute Unified Device Architecture***
---
## reference:
### Professional CUDA C Programming
- [1](http://www.hds.bme.hu/~fhegedus/C++/Professional%20CUDA%20C%20Programming.pdf)
### tutorial:
- [1](https://www.ptt.cc/bbs/C_and_CPP/M.1224075534.A.C49.html)
- [2](https://ppt.cc/dx2Z)
- [3](https://www.ptt.cc/bbs/VideoCard/M.1223477316.A.1F8.html)
- [4](https://www.ptt.cc/bbs/C_and_CPP/M.1224075621.A.9AA.html)
- [5](https://www.ptt.cc/bbs/C_and_CPP/M.1224075634.A.1B1.html)
- [6](https://www.ptt.cc/bbs/C_and_CPP/M.1224674646.A.F5F.html)
- [7](https://ppt.cc/2s22)
- [8](https://www.ptt.cc/bbs/C_and_CPP/M.1225912248.A.3DF.html)
- [9](https://ppt.cc/9G_E)
- [10](https://www.ptt.cc/bbs/C_and_CPP/M.1226502649.A.87B.html)
- [11](https://www.ptt.cc/bbs/C_and_CPP/M.1227119415.A.BB5.html)
- [12](http://www.cnblogs.com/1024incn/tag/CUDA/)
- [13](https://www.ptt.cc/bbs/VideoCard/M.1228930736.A.779.html)
- [14](https://www.ptt.cc/bbs/VideoCard/M.1231036765.A.649.html)
- [15](https://www.ptt.cc/bbs/C_and_CPP/M.1233304368.A.013.html)

### basic:
- [1](http://www.jianshu.com/p/0afb1305b1ae)
- [2](http://blog.csdn.net/csgxy123/article/details/9704461)
- [3](http://www.cnblogs.com/1024incn/tag/CUDA/)
- [4](https://www.slideshare.net/aj0612/mosutgpucoding-cuda)
- [5](https://chenrudan.github.io/blog/2015/07/22/cudastream.html)

### efficiency:
- [1](https://stackoverflow.com/questions/10460742/how-do-cuda-blocks-warps-threads-map-onto-cuda-cores)
- [2](https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s)
- [3](https://malagastockholm.wordpress.com/2013/01/13/optimizing-cuda-warps-threads-and-blocks/)

### image: 
- [1](https://github.com/jamolnng/OpenCL-CUDA-Tutorials/blob/master/CUDA/Tutorial%202%20-%20CUDA%20load%20image/kernel.cu)
- [2](https://stackoverflow.com/questions/2392250/understanding-cuda-grid-dimensions-block-dimensions-and-threads-organization-s)



## GPU Spec

|            |    GTX1080   |   GTX1080Ti  | Tesla P40   | Tesla P4 |
| ---------  | ------------ | ------------ | ---------- | -------- |
| GPU Name   |     GP104    |     GP102    |   GP102    |  GP104   |
| Process    |     16nm     |     16nm     |    16nm    |   16nm   |
| Memory     |  8GB GDDR5X  |  11GB GDDR5X |24GB GDDR5X |8GB GDDR5X|
|CUDA Cores  |     2560     |     3584     |    3840    |   2560   |
|SMs	       |       20     |      28    	 |     30    	|    20    |
|Cores/SM    |	     128    |	    128      |	    128   |	   128   |
|Memory Clock|   10008 MHz	|   11008 MHz	 | 10008 MHz	| 10008 MHz|
|Memory Interface(bus)|  256-bit|  352-bit |  384-bit 	|  256-bit |
|Memory Bandwidth|	320GB/s	|   484 GB/s	 |  347 GB/s	| 192 GB/s |
| Base Clock |   1607MHz    |	   1480 MHz  |	 1303 MHz	| 810 MHz  |
|Compute|9TFLOPS(8873GFLOPS)|	11.5TFLOPS   |9TFLOPS(11,758GFLOPS)|5.5TFLOPS(5,443GFLOPS)|
|Architecture|   Pascal     |	   Pascal    |	 Pascal 	 | Pascal   |
|Threads/Warp|	   32        |       32     |  	 32     |   32     |
|MaxThreadDim|(1024, 1024, 64)|            |          	|          |
|MaxGridSize |(2^32/2, 65535, 65535)| |     |     |
|MaxThreadPerBlock|  1024	  |     	   |    	|     |
|PerBlockSharedMem|49152 byte  |     	   |    	|     |
|PerBlockRegistMem|65536 byte  |     	   |     |     |
---


## CUDA Files
| |File Prefix Description|
| ------ | ------ |
|.cu | CUDA source file, containing host code and device functions|
|.c | C source file|
|.cc, .cxx, .cpp | C++ source file|
|.gpu | GPU intermediate file|
|.ptx | PTX intermediate assembly file|
|.o, .obj | Object file|
|.a, .lib | Library file|
|.so| Shared object file|
---

### Architecture Feature
#### Real Architecture
| Architecture  | Feature |
| --- | --- |
|sm_20| Basic features + Fermi support|
|sm_30, sm_32 | + Kepler support + Unified memory programming|
|sm_35 | + Dynamic parallelism support|
|sm_50, sm_52, sm_53 | + Maxwell support|
|sm_60, sm_61 | + Pascal support|

#### Virtul Architecture 
| Architecture  | Feature |
| --- | --- |
|compute_20 |Basic features + Fermi support|
|compute_30, compute_32 | + Kepler support + Unified memory programming|
|compute_35 | + Dynamic parallelism support|
|compute_50, compute_52, compute_53 | + Maxwell support|
|compute_60, compute_61 | + Pascal support|

***Supported on CUDA 7 and later*** <br />
* SM20 – Older cards such as GeForce GT630 <br />
* SM30 – Kepler architecture (generic – Tesla K40/K80)
Adds support for unified memory programming <br />
* SM35 – More specific Tesla K40 
Adds support for dynamic parallelism. Shows no real benefit over SM30 in my experience. <br />
* SM37 – More specific Tesla K80 
Adds a few more registers. Shows no real benefit over SM30 in my experience <br />
* SM50 – Tesla/Quadro M series <br />
* SM52 – Quadro M6000 , GTX 980/Titan <br />
* SM53 – Tegra TX1 / Tegra X1 <br />

***Supported on CUDA 8 and later*** <br />
* SM60 – GP100/Pascal P100 – DGX-1 (Generic Pascal) <br />
* SM61 – GTX 1080, 1070, 1060, Titan Xp, Tesla P40, Tesla P4 <br />
* SM62 – Probably Drive-PX2 <br />

***Supported on CUDA 9 and later***  <br />
* SM70 – Tesla V100 <br />

---

## CUDA Architecture
### Equipment
* **主機 (host)**  ：顯示卡那台PC。
* **裝置 (device)**：顯示卡。
* **核心 (kernel)**：顯示卡上執行的程式碼區段。


### Hardware
* **SP(Streaming Process)**: 最基本的處理單元，一個SP可執行一個thread。 <br />
* **SM(Streaming Multiprocessor)**: 由多個SP加上一些資源而組成，每個SM所擁有之SP數量依據不同GPU架構而不同，Kepler=128,Maxwell=128,Pascal=128，任一GPU則有多個SM，軟體定義上SP是平行運算，但物理實現上並非所有SP能夠同時平行運算，有些會處於掛起、就緒等其他狀態。任一SM只能同時執行一個block，如GTX1080有20個SM，故只能同時執行20個blocks。在執行的過程中block之間互不干擾，執行順序是隨機的，故多餘之線程塊blocks則呈現queue狀態排隊進入SM。  <br />


### Software
* **thread**: 一個CUDA的併行程序(kernel)會以多個threads來執行，為3D結構。 <br />
* **block**: 由多個threads所組成，同一block中之threads可以同步，也可利用shared memory來共享資料，為3D結構，設計blockDim時候盡量以32(warp)之倍數來設計，官方建議至少為64、128、256這些數字。 <br />
* **grid**: 由多個blocks所組成，只能為2D結構。 <br />
* **warp**: 32個threads組成一個warp，warp是調度和運行的基本單元。warp中所有threads並行的執行相同的指令，一個warp需要佔用一個SM運行，多個warps需要輪流進入SM，故只能有一個warp正被執行。 <br />
  * **active warp**: 指已分配給SM的warp，且該warp所需資源如暫存器也已分配，一個SM最多有128/32=4個warp(GTX1080)。 <br />
       ===**Selected warp**:被調度器warp schedulers挑選中送去執行之active warp <br />
       ===**Stalled warp**:還沒準備好要去執行之active warp <br />
       ===**Eligible warp**:還未被選中但已經準備好要執行之active warp <br />
  * **resident thread**: 為一個正在SM裡同時執行之warp數，一個GPU上resident thread最多只(SM)x32個。 <br />


### Memory
* **Registers**: 暫存器-(thread) <br />
* **Shared Memory**: 共享記憶體，任一block皆有獨立之共享記憶體-(block) <br />
* **Host Memory**: 主機記憶體-(RAM) <br />
* **Device Memory**: 裝置記憶體，又稱global memory對GPU裡所有線程皆可讀取-(GPU) <br />
* **Local Memory**: 本地記憶體，任一線程皆有獨立之本地記憶體-(thread) <br />

| name       |   position   | read/write speed |
| ---------  | ------------ | ------------ |
|  Registers |    GPU   |     immediately    |  
|  Shared Memory  |   GPU      |   4 cycles   |
|  Host Memory    |   PC Board | (PCI-E) Slow |  
|  Device Memory  |     GPU    |400-600 cycles|  


### CUDA API
* **cudaMalloc()**: 配置device記憶體 <br />
* **cudaFree()**: 釋放device記憶體 <br />
* **cudaMemcpy()**: 記憶體複製，注意cudaMemcpy函數是同步的，將等待kernel中所有線程都完成了執行，再執行數據的拷貝<br />
* **cudaGetErrorstring()**: 錯誤字串解釋 <br />
* **~~cudaThreadSynchronize() 舊的~~** / **cudaDeviceSynchronize()**:is a _host_ function，同步化，在運行內核時須用到，內核中所有線程式不同步的，直到所有任務計算完畢時須同步，且CUDA API和host代碼是異步的，cudaDeviceSynchronize可以用來強制停住host(CUP)端等待device(CUDA)端中所有的線程都完成執行。 <br />

* **__syncthreads()**:is a _device_ function，來進行區塊內的執行緒同步，避免資料時序上的問題(來自不同的threads)，常與共享記憶體使用，用來隔開共享記憶體之**寫入週期**和**讀取週期**，避免WAR(write after read)。 <br />

* **cudaStreamSynchronize()**: 與cudaDeviceSynchronize()類似功能，但函式帶有一參數Stream。<br />


* **kernel**: one kernel = one grid   <br />
   **宣告**: <br />
      ___ global ___ void kernel_name(type1 arg1, type2 arg2, ...){ <br />
                 函式內容 <br />
      }; <br />
   **呼叫**: <br />
      kernel<<< blocks, threads >>>( arg1, arg2, ... ); <br />
      
   ```C++
      任務(kernel)
        |
        |                  +--> 區塊block(排or班) +--> 執行緒(小兵)
        |                  |                      +--> 執行緒(小兵)
        |                  |                      +--> 執行緒(小兵) 
        |                  |                      +--> 執行緒(小兵)
        |                  |
        +--> 網格grid(連隊) +--> 區塊block(排or班) +--> 執行緒(小兵)
                           |                      +--> 執行緒(小兵)
                           |                      +--> 執行緒(小兵)
                           |                      +--> 執行緒(小兵)
                           |
                           +--> 區塊block(排or班) +--> 執行緒(小兵)
                                                  +--> 執行緒(小兵)
                                                  +--> 執行緒(小兵)
                                                  +--> 執行緒(小兵)
   ```

### CUDA only read variable
#### dim3 = uint3
struct uint3{ <br />
unsigned int x, y, z; <br />
} <br />

**uint3 only be used on .cu file,but dim3 can be used on .cpp/.cu file**

* **threadIdx:** thread index
* **blockIdx:** block index
* **blockDim:**  (threadsPerBlock)
* **gridDim:**  (numBlocks)

---

### CUDA Matrix sum

***host***
  ```C++
float *ia=A;
float *ib=B;
float *ic=C;
for(int iy=0; iy < ny; iy++){
    for(int ix=0; ix < nx; ix++){
        ic[ix]=ia[ix]+ib[ix];
    }
    ia+=nx; ib+=nx; ic+=nx;
}
  ```

***device***
  
  #### 2D Grid 2D Block
  ```C++
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int Idx = nx*iy + ix;
  if(ix < nx && iy < ny){
      MatC[Idx]=MatA[Idx]+MatB[Idx];
  }
  ```
  #### 2D Grid 1D Block
  ```C++
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
  unsigned int iy = threadIdx.y;
  unsigned int Idx = nx*iy + ix;
  if(ix < nx && iy < ny){
      MatC[Idx]=MatA[Idx]+MatB[Idx];
  }
  ```
  #### 1D Grid 1D Block
  ```C++
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x; 
  if(ix < nx){
      for(int iy=0; iy < ny; iy++){
      int idx = nx*iy + ix;
      MatC[idx]=MatA[idx]+MatB[idx];
      }
  }
  
  ```
---
### Warp Diveragence
  GPGPU 的分支指令(branch, 條件判斷if else)是比較弱的, 理由是其執行上是以 warp 為單位, 當 32 個執行緒條件不同時,
  就會導致部份的執行緒停擺, 等到下一執行週期再執行, 這種變成循序執行的動作稱為發散 (divergence), 會造成這段指令執
  行需要兩倍時間而拖慢效能.這種去除條件判斷和迴圈的程式設計法我稱為「乾式(dry)」的設計法,因為缺乏流程控制(flow control),
  其缺點為在比較複雜的程式中容易失去彈性,而且必需付出計算資料位址的額外成本 (每個執行緒都必需計算一次).<br />
**發生點**:同個warp內之thread跑在不同程序裡(如if/else判斷式)<br />
**解決法**:將會分支程序內之計算粒度調整為warp大小倍數<br />

  ```C++
 __global__ void mathKernel2( void ) {
     int tid = blockIdx.x * blockDim.x + threadIdx.x;
     float a, b;
    a = b = 0.0f ;
     if ((tid / warpSize) % 2 == 0 ) {
        a = 100.0f ;
    } else {
        b = 200.0f ;
    }
    c[tid] = a + b;
} 
  ```

#### nvprof度量性能:

  ```C++
  $ nvprof --metrics branch_efficiency ./XXXX...
  ```
  
#### nvprof計算branch/divergent_branch數量:
  ```C++
  $ nvprof --events branch,divergent_branch ./XXXX...
  ```

---
### Latency Hiding 延遲隱藏

* **Arithmetic instruction:** 為每個SM端之延遲計算 <br />
       latency : 10-20 cycles for arithmetic operations <br />
       Number of Required Warps =Latency(cycles) X Throughput <br />
       throughput定義為每個SM每個cycle的操作數目 <br />
        
* **Memory instruction:** 為記憶體搬移之延遲計算 <br />
       latency : 400-800 cycles for arithmetic operations <br />
       Number of Required Warps =Latency(cycles) X Memory Frequency(bandwidth) <br />

### Occupancy
- [CUDA Calculator1](https://devtalk.nvidia.com/default/topic/368105/cuda-occupancy-calculator-helps-pick-optimal-thread-block-size/)
- [CUDA Calculator2](http://lxkarthi.github.io/cuda-calculator/)

    **Occupancy=active warps / maxmum warps** <br />
    maxmum warps可用maxThreadsPerMultiProcessor取得 <br />
        <br />
    **grid和block的配置準則**：
    * 保證block中thrad數目是32的倍數。 <br />
    * 避免block太小：每個blcok最少128或256個thread。 <br />
    * 根據kernel需要的資源調整block。 <br />
    * 保證block的數目遠大於SM的數目。 <br />
    * 多做實驗來挖掘出最好的配置。 <br />
Occupancy專注於每個SM中可以並行的thread或者warp的數目。不管怎樣，Occupancy不是唯一的性能指標，Occupancy達到當某個值是，再做優化就可能不在有效果了，還有許多其它的指標需要調節，我們會在之後的博文繼續探討。 <br />

---

### Exposing Parallelism

#### nvprof計算每個SM在每個cycle能夠達到的最大active warp數目佔總warp的比例  (單位 無 ):
  ```C++
  $ nvprof --metrics achieved_occupancy ./XXXX...
  ```
<br />

#### nvprof計算memory的throughput (單位 GB/s ):
  ```C++
  $ nvprof --metrics gld_throughput ./XXXX...
  ```
<br />

#### 使用nvprof的gld_efficiency來度量load efficiency (單位 % ):
  ```C++
  $ nvprof --metrics gld_efficiency ./XXXX...
  ```
該metric參數是指我們確切需要的global load throughput與實際得到global load memory的比值。這個metric參數可以讓我們知道，APP的load操作利用device memory bandwidth的程度

<br />

#### Conclusion:
現在可以看出，最佳配置既不是擁有最高achieved Occupancy也不是最高load throughput的。所以不存在唯一metric來優化計算性能，我麼需要從眾多metric中尋求一個平衡。 <br />

- 在大多數情形下，並不存在唯一的metric可以精確的優化性能。
- 哪個metric或者event對性能的影響大多是由kernel具體的代碼決定的。
- 在眾多相關的metric和event中尋求一個平衡。
- Grid/blcok heuristics（啟發） 為調節性能提供了不錯的切入點。
<br />

---

### Parallel Reduction
- 將輸入數組切割成很多小的塊。
- 用thread來計算每個塊的和。
- 對這些塊的结果再求和得最终结果。


**Original function**
 ```C++
  int sum = 0;
  for (int i = 0; i < N; i++)
    sum += array[i];
  ```
<br />

**Neighbored pair：每次迭代都是相鄰兩個元素求和。**

```C++
__global__ void reduceNeighbored(int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x;
    // extern __shared__ int sdata[];   //we can also use share memory to load data
    
    // boundary check
    if (idx >= n) return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        if ((tid % (2 * stride)) == 0) {
            idata[tid] += idata[tid + stride];
            //sdata[tid] += sdata[tid+stride];  //if using share memory
        }
        // synchronize within block
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
    //if (tid == 0) g_odata[blockIdx.x] = sdata[0]; //if using share memory
}        
```
<br />


```C++
__global__ void reduceNeighboredLess (int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x*blockDim.x;
    
    // boundary check
    if(idx >= n) return;
    
    // in-place reduction in global memory
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
        // convert tid into local array index
        int index = 2 * stride * tid;
        if (index < blockDim.x) {
            idata[index] += idata[index + stride];
        }    
        // synchronize within threadblock
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}  

```

我們也可以使用nvprof的inst_per_warp參數來查看每个warp上執行的指令數目的平均值。<br/>
```C++
$ nvprof --metrics inst_per_warp ./xxx
```
<br />


**Interleaved pair：按一定跨度配對各個元素。**<br />
```C++
__global__ void reduceInterleaved (int *g_idata, int *g_odata, unsigned int n) {
// set thread ID
unsigned int tid = threadIdx.x;
unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

// convert global data pointer to the local pointer of this block
int *idata = g_idata + blockIdx.x * blockDim.x;

// boundary check
if(idx >= n) return;

// in-place reduction in global memory
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
        idata[tid] += idata[tid + stride];
    }
    __syncthreads();
}

// write result for this block to global mem
if (tid == 0) g_odata[blockIdx.x] = idata[0];
}
```
<br />

### UNrolling Loops
```C++
for (int i = 0; i < 100; i++) {
    a[i] = b[i] + c[i];
}
```

```C++
for (int i = 0; i < 100; i += 2) {
    a[i] = b[i] + c[i];
    a[i+1] = b[i+1] + c[i+1];
} 
```

<br />

每個block處理一部分數據，我們給這數據起名data block 下面的代碼是reduceInterleaved的修正版本，每個block，都是以兩個data block作為源數據進行操作每個thread作用於多個data block，並且從每個data block中取出一個元素處理。 <br />
```C++
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;
```
**|-block1-| |-block2-| |-block3-| |-block4-| |-block5-| |-block6-|** <br />
-----↑_block size_↑-----        


每個thread從相鄰的data block中取數據，這一步實際上就是將兩個data block規約成一個。 <br />

以Interleaved pair範例改編如下: <br />
```C++
__global__ void reduceUnrolling2 (int *g_idata, int *g_odata, unsigned int n) {
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2 data blocks
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];
    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            idata[tid] += idata[tid + stride];
        }
        // synchronize within threadblock
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}   
```
<br />

---

### Streams and Events

- **Kernel level** : 一個kernel或者一個task由許多thread並行的執行在GPU上。<br />
- **Grid level** : Grid level是指多個kernel在一個device上同時執行(stream概念)。 <br />

Cuda stream是指一堆異步的cuda操作，他們按照host代碼調用的順序執行在device上，在許多情況下，花費在執行kernel上的時間要比傳輸數據多得多，所以很容易想到將cpu和gpu之間的溝通時間隱藏在其他kernel執行過程中，我們可以將數據傳輸和kernel執行放在不同的stream中來實現此功能。Stream可以用來實現pipeline和雙buffer（front-back）渲染，從軟件角度來看，不同stream中的不同操作可以並行執行，但是硬件角度卻不一定如此。這依賴於PCIe鏈接或者每個SM可獲得的資源，不同的stream仍然需要等待別的stream來完成執行。<br />

**一般cudaMemcpy為host和device端同步傳遞資料，其異步版本的cudaMemcpy如下：**
```C++
cudaError_t cudaMemcpyAsync( void * dst, const  void * src, size_t count,cudaMemcpyKind kind, cudaStream_t stream = 0 );
```
<br />

**如果要聲明一個新的stream則使用下面的API定義一個：**
```C++
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
```
<br />

**當執行一次異步數據傳輸時，我們必須使用pinned（或者non-pageable）memory。Pinned memory的分配如下:**
```C++
cudaError_t cudaMallocHost( void ** ptr, size_t size);
cudaError_t cudaHostAlloc( void **pHost, size_t size, unsigned int flags);
```
<br />

**在執行kernel時要想設置stream的話，也是很簡單的，同樣只要加一個stream參數就好：**
```C++
kernel_name<<<grid, block, sharedMemSize, stream >>>(argument list);
```
<br />

**宣告創建與銷毀stream:**
```C++
cudaStream_t stream;
cudaStreamCreate(& stream);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
```
<br />

**由於所有stram的執行都是異步的，就需要一些API在必要的時候做同步操作：**
```C++
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
```
**第一個會強制host阻塞等待，直至stream中所有操作完成為止；第二個會檢查stream中的操作是否全部完成，即使有操作沒完成也不會阻塞host。** <br/>
<br />

**stream可以有優先級的屬性：**
```C++
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
```
<br />

以下代碼使用了三個stream，數據傳輸和kernel運算都被分配在了這幾個並發的stream中。
```C++
for ( int i = 0 ; i < nStreams; i++ ) {
     int offset = i * bytesPerStream;
    cudaMemcpyAsync( &d_a[offset], &a [offset], bytePerStream, streams[i]);
    kernel <<grid, block, 0 , streams[i]>>(& d_a[offset]);
    cudaMemcpyAsync( &a[offset], & d_a[offset], bytesPerStream, streams[i]);
}

for ( int i = 0 ; i < nStreams; i++ ) {
    cudaStreamSynchronize(streams[i]);
}
```

### Cuda Events

Event是stream相關的一個重要概念，其用來標記strean執行過程的某個特定的點。其主要用途是：
- 同步stream執行
- 操控device運行步調
只有當該event標記的stream位置的所有操作都被執行完畢，該event才算完成。<br />

```C++
//  宣告 
cudaEvent_t event ;
 // 創建 
cudaError_t cudaEventCreate(cudaEvent_t* event );
 // 銷毀 
cudaError_t cudaEventDestroy(cudaEvent_t event );
```
<br/>


```C++
下面函數將event關聯到指定stream:
cudaError_t cudaEventRecord(cudaEvent_t event , cudaStream_t stream = 0 );

等待event會阻塞調用host線程，同步操作調用下面的函數：
cudaError_t cudaEventSynchronize(cudaEvent_t event );

我們同時可以使用下面的API來測試event是否完成，該函數不會阻塞host：
cudaError_t cudaEventQuery(cudaEvent_t event );

還有專門的API可以度量兩個event之間的時間間隔：
cudaError_t cudaEventElapsedTime( float * ms, cudaEvent_t start, cudaEvent_t stop);
```
<br/>

下面代碼簡單展示瞭如何使用event來度量時間：
```C++
// create two events 
cudaEvent_t start, stop;
cudaEventCreate( & start);
cudaEventCreate( & stop);

 // record start event on the default stream 
cudaEventRecord(start);

// execute kernel 
kernel<<<grid, block>>> (arguments);

// record stop event on the default stream 
cudaEventRecord(stop );

// wait until the stop event completes 
cudaEventSynchronize(stop);

// calculate the elapsed time between two events 
float time;
cudaEventElapsedTime( & time, start, stop);

// clean up the two events 
cudaEventDestroy(start);
cudaEventDestroy(stop);
```

### Stream Synchronization
以stream來說分成異步和同步兩種，同步異步是針對host來講的，異步Stream又分為阻塞和非阻塞兩種，阻塞非阻塞是異步stream針對同步stream來講的: <br/>
➤ Synchronous streams (the NULL/default stream) <br/>
➤ Asynchronous streams (non-NULL streams/non-default stream ) ----------->  Blocking streams / Non-blocking streams <br />

除了通過cudaStreamCreate生成的阻塞stream外，我們還可以通過下面的API配置生成非阻塞Non-blocking stream：
```C++
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
 
 // flag為以下兩種，默認為第一種，非阻塞便是第二種。
  cudaStreamDefault : default stream creation flag (blocking)
  cudaStreamNonBlocking : asynchronous stream creation flag (non -blocking)
```

### Implicit Synchronization 隱式同步
隱式同步我們也了解過，比如cudaMemcpy就會隱式的同步device和host，因為該函數同步作用只是數據傳輸的副作用，所以稱為隱式。了解這些隱式同步是很中要的，因為不經意的調用這樣一個函數可能會導致性能急劇降低。

### Explicit Synchronization 顯式同步
顯式同步API有：<br />
- cudaDeviceSynchronize
- cudaStreamSynchronize
- cudaEventSynchronize


從grid level來看顯式同步方式，有如下幾種：<br />
- Synchronizing the device
- Synchronizing a stream
- Synchronizing an event in a stream
- Synchronizing across streams using an event

我們可以使用之前提到過的**cudaDeviceSynchronize**來同步該device上的所有操作。該函數會導致host等待所有device上的運算或者數據傳輸操作完成。顯而易見，該函數是個heavyweight的函數，我們應該盡量減少這類函數的使用。<br/>
通過使用**cudaStreamSynchronize**可以使host等待特定stream中的操作全部完成或者使用非阻塞版本的**cudaStreamQuery**來測試是否完成。<br/>
Cuda event可以用來實現更細粒度的阻塞和同步，相關函數為**cudaEventSynchronize**和**cudaEventSynchronize**，用法類似stream相關的函數。此外，**cudaStreamWaitEvent**提供了一種靈活的方式來引入stream之間的依賴關係：
```C++
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event );
```
該函數會指定該stream等待特定的event，該event可以關聯到相同或者不同的stream，對於不同stream的情況。<br/>
Stream2會等待stream1中的event完成後繼續執行。<br/>
<br />

**Event的配置可用下面函數：**
```C++
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event , unsigned int flags);
cudaEventDefault
cudaEventBlockingSync
cudaEventDisableTiming
cudaEventInterprocess
```
<br/>

---

### CUDA Memory Model
- **Registers** : 寄存器是GPU最快的memory，kernel中沒有什麼特殊聲明的自動變量都是放在寄存器中的
- **Local memory** : 如果register不夠用了，那麼就會使用local memory來代替這部分寄存器空間，local memory有很高的latency和較低的bandwidth
- **Shared memory** : 用__shared__修飾符修飾的變量存放在shared memory，因為shared memory是on-chip的，他相比localMemory和global memory來說，擁有高的多bandwidth和低很多的latency，shared memory是以block為單位分配的
- **Constant memory** : 當一個warp中所有thread都從同一個Memory地址讀取數據時，constant Memory表現最好。例如，計算公式中的係數
- **Texture memory** : texture Memory是針對2D空間局部性的優化策略，所以thread要獲取2D數據就可以使用texture Memory來達到很高的性能
- **Global memory** : global Memory是空間最大，latency最高，GPU最基礎的memory

### CUDA Variable Declaration Summary
|  QUALIFIER    |  VARIABLE NAME |   MEMORY      |    SCOPE    |   LIFESPAN   |
| ---------     |  ------------  |  ------------ |  ---------- |  ----------  |
|               |  float var     |   Register    |   Thread    | Thread       |
|               |  float/var[100]|   Local       |   Thread    | Thread       |
| __ shared __  |  float var †   |   Shared      |   Block     | Block        |
| __ device __  |  float var †   |   Global      |   Global    | Application  |
| __ constant __|  float var †   |   Constant    |   Global    | Application  |

<br/>

### Memory Allocation and Deallocation
在分配global Memory時，最常用的就是下面這個了：
```C++
cudaError_t cudaMalloc(void **devPtr, size_t count);
```
如果分配出錯則返回cudaErrorMemoryAllocation。分配成功後，就得對該地址初始化值，要嘛從host調用cudaMemcpy賦值，要嘛調用下面的API初始化：
```C++
cudaError_t cudaMemset(void *devPtr, int value, size_t count);
```
釋放資源就是：
```C++
cudaError_t cudaFree(void *devPtr);
```

### Memory Transfer
一旦global Memory分配好後，如果不用cudaMemset就得用下面這個：
```C++
cudaError_t cudaMemcpy(void *dst, const void *src, size_t count,enum cudaMemcpyKind kind);
```
kind就是下面這幾種：
```C++
cudaMemcpyHostToHost
cudaMemcpyHostToDevice
cudaMemcpyDeviceToHost
cudaMemcpyDeviceToDevice
```

### Pinned Memory
Host Memory的分配默認情況下是pageable的，也就是說，我們要承受因pagefault導致的操作，，這個操作要將host virtual Memory的數據轉移到由OS決定的不物理位置，當將pageable host Memory數據送到device時，CUDA驅動會首先分配一個臨時的page-locked或者pinned host Memory，並將host的數據放到這個臨時空間裡。然後GPU從這個所謂的pinned Memory中獲取數據。<br/>

我們也可以顯式的直接使用pinned Memory，如下：
```C++
cudaError_t cudaMallocHost(void **devPtr, size_t count);
```
由於pinned Memory能夠被device直接訪問（不是指不通過PCIE了，而是相對左圖我們少了pageable Memory到pinned Memory這一步），所以他比pageable Memory具有相當高的讀寫帶寬，當然像這種東西依然不能過度使用，因為這會降低pageable Memory的數量，影響整個虛擬存儲性能<br/>

Pinned Memory的釋放也比較特殊：
```C++
cudaError_t cudaFreeHost(void *ptr);
```
將許多小的傳輸合併到一次大的數據傳輸，並使用pinned Memory將降低很大的傳輸消耗。 <br/>

### Zero-Copy Memory
Zero-copy本身實質就是pinned memory並且被映射到了device的地址空間。下面是他的分配API：
```C++
cudaError_t cudaHostAlloc(void **pHost, size_t count, unsigned int flags);
```

flags是保留參數，留待將來使用，目前必須設置為零。
其資源釋放當然也是cudaFreeHost，至於flag則是下面幾個選項：
```C++
cudaHostAllocDefault
cudaHostAllocPortable
cudaHostAllocWriteCombined
cudaHostAllocMapped
```
當使用**cudaHostAllocDefault**時，cudaHostAlloc和cudaMallocHost等價。<br/>
**cudaHostAllocWriteCombined**是在特殊系統配置情況下使用的，這塊pinned memory在PCIE上的傳輸更快，但是對於host自己來說，卻沒什麼效率。<br/>
最常用的是**cudaHostAllocMapped**，就是返回一個標準的zero-copy。<br/>

可以用下面的API來獲取device端的地址：
```C++
cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
```

下面一段代買是比較頻繁讀寫情況下，zero-copy的表現：
```C++
int main( int argc, char ** argv) {
 // part 0: set up device and array
 // set up device 
int dev = 0 ;
cudaSetDevice(dev);
// get device properties 
cudaDeviceProp deviceProp;
cudaGetDeviceProperties( & deviceProp, dev);
 // check if support mapped memory 
if (! deviceProp.canMapHostMemory) {
printf( " Device %d does not support mapping CPU host memory!\n " , dev);
cudaDeviceReset();
exit(EXIT_SUCCESS);
}
printf( " Using Device %d: %s " , dev, deviceProp.name);
 // set up date size of vectors 
int ipower = 10 ;
 if (argc> 1 ) ipower = atoi(argv[ 1 ]);
 int nElem = 1 << ipower;
size_t nBytes = nElem * sizeof ( float );
 if (ipower < 18 ) {
printf( " Vector size %d power %d nbytes %3.0f KB\n " , nElem,\
ipower,( float )nBytes/( 1024.0f ));
} else {
printf( " Vector size %d power %d nbytes %3.0f MB\n " , nElem,\
ipower,( float )nBytes/( 1024.0f * 1024.0f ));
}
// part 1: using device memory
 // malloc host memory 
float *h_A, *h_B, *hostRef, * gpuRef;
h_A = ( float *) malloc (nBytes);
h_B = ( float *) malloc (nBytes);
hostRef = ( float *) malloc (nBytes);
gpuRef = ( float *) malloc (nBytes);
 // initialize data at host side 
initialData(h_A, nElem);
initialData(h_B, nElem);
memset(hostRef, 0 , nBytes);
memset(gpuRef, 0 , nBytes);
 // add vector at host side for result checks 
sumArraysOnHost(h_A, h_B, hostRef, nElem);
 // malloc device global memory 
float *d_A, *d_B, * d_C;
cudaMalloc(( float **)& d_A, nBytes);
cudaMalloc(( float **)& d_B, nBytes);
cudaMalloc(( float **)& d_C, nBytes);
 // transfer data from host to device 
cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);
// set up execution configuration 
int iLen = 512 ;
dim3 block (iLen);
dim3 grid ((nElem +block.x- 1 )/ block.x);
 // invoke kernel at host side 
sumArrays <<<grid, block>>> (d_A, d_B, d_C, nElem);
 // copy kernel result back to host side 
cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
 // check device results 
checkResult(hostRef, gpuRef, nElem);
 // free device global memory 
cudaFree(d_A);
cudaFree(d_B);
free (h_A);
 free (h_B);
 // part 2: using zerocopy memory for array A and B
 // allocate zerocpy memory 
unsigned int flags = cudaHostAllocMapped;
cudaHostAlloc(( void **)& h_A, nBytes, flags);
cudaHostAlloc(( void **)& h_B, nBytes, flags);
 // initialize data at host side 
initialData(h_A, nElem);
initialData(h_B, nElem);
memset(hostRef, 0 , nBytes);
memset(gpuRef, 0 , nBytes);
 // pass the pointer to device 
cudaHostGetDevicePointer(( void **)&d_A, ( void *)h_A, 0 );
cudaHostGetDevicePointer(( void **)&d_B, ( void *)h_B, 0 );
 // add at host side for result checks 
sumArraysOnHost(h_A, h_B, hostRef, nElem);
 // execute kernel with zero copy memory 
sumArraysZeroCopy <<< grid, block>>> (d_A, d_B, d_C, nElem);
 // copy kernel result back to host side 
cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
 // check device results 
checkResult(hostRef, gpuRef, nElem);
 // free memory 
cudaFree(d_C);
cudaFreeHost(h_A);
cudaFreeHost(h_B);
free (hostRef);
 free (gpuRef);
 // reset device 
cudaDeviceReset();
 return EXIT_SUCCESS;
}
```
### Unified Virtual Addressing(UVA)
在使用UVA的情況下，CPU和GPU使用同一塊連續的地址空間。<br/>
使用UVA之後，就沒必要來獲取device的映射地址了，直接使用一個地址就可以，如下代碼所示：
```C++
// allocate zero-copy memory at the host side 
cudaHostAlloc(( void **)& h_A, nBytes, cudaHostAllocMapped);
cudaHostAlloc(( void **)& h_B, nBytes, cudaHostAllocMapped);

 // initialize data at the host side 
initialData(h_A, nElem);
initialData(h_B, nElem);

// invoke the kernel with zero-copy memory 
sumArraysZeroCopy<<<grid, block>>>(h_A, h_B, d_C, nElem);
```
<br/>

---

## CUDA Image Processing Example

### reference:
- [1_opencv+cuda](http://lps-683.iteye.com/blog/2282079)


**高斯均值濾波**
```C++
template <typename T> __global__ void MeanFilterCUDA(T* pInput, T* pOutput, int nKernelSize, int nWidth, int nHeight)
{
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    int j = blockIdx.y;
    int pos = j*nWidth + i;  //pixel index
    
    if( i>0 && i < nWidth-1 && j > 0 && j < nHeight-1)  //process scope
    {
        float temp += pInput[pos]; 
        temp += pInput[pos+1]; 
        temp += pInput[pos-1]; 
        temp += pInput[pos - nWidth]; 
        temp += pInput[pos - nWidth + 1]; 
        temp += pInput[pos - nWidth - 1]; 
        temp += pInput[pos + nWidth]; 
        temp += pInput[pos + nWidth + 1]; 
        temp += pInput[pos + nWidth - 1];
        pOutput[pos] = (T)(temp/nKernelSize);
    }
    else
    {
        pOutput[pos]=pInput[pos];    
    }
}
```

```C++
dim3 block(256,1,1);
dim3 grid(nWidth+255/block.x, nHeight, 1);
MeanFilterCUDA<<<grid, block>>>(dataIn, dataOut, kernelsize, width, height);
```
<br/>

#### 圖像金字塔
這裡的向下與向上採樣，是對圖像的尺寸而言的（和金字塔的方向相反），向上就是圖像尺寸加倍，向下就是圖像尺寸減半。 <br/>
- 對圖像向上採樣：pyrUp函數 [拉普拉斯金字塔(Laplacianpyramid)] **(放大)**



- 對圖像向下採樣：pyrDown函數 [高斯金字塔 ( Gaussianpyramid)] **(縮小)**
首先將原圖像作為最底層圖像G0（高斯金字塔的第0層），利用高斯核（5x5)對其進行卷積，然後對卷積後的圖像進行下採樣（去除偶數行和列）得到上一層圖像G1，將此圖像作為輸入，重複卷積和下採樣操作得到更上一層圖像，反覆跌代多次，形成一個金字塔形的圖像數據結構，即高斯金字塔



