# GPU_CUDA
***CUDA=Compute Unified Device Architecture***
---
## reference:
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

  ```
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

  ```
  $ nvprof --metrics branch_efficiency ./XXXX...
  ```
  
#### nvprof計算branch/divergent_branch數量:
  ```
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
- [CUDA Calculator](https://devtalk.nvidia.com/default/topic/368105/cuda-occupancy-calculator-helps-pick-optimal-thread-block-size/)


