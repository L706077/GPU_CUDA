# GPU_CUDA

## GPU Spec

|            |    GTX1080   |   GTX1080Ti  | Tesla P4   | Tesla P4 |
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
|PerBlockSharedMem|49152kb  |     	   |    	|     |
|PerBlockRegistMem|65536kb  |     	   |     |     |
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
|Real Architecture| |
| --- | --- |
| Architecture  | Feature |
|sm_20| Basic features + Fermi support|
|sm_30 and sm_32 | + Kepler support + Unified memory programming|
|sm_35 | + Dynamic parallelism support
|sm_50, sm_52, and sm_53 | + Maxwell support|
|Virtul Architecture| |
| --- | --- |
| Architecture  | Feature |
|compute_20 |Basic features + Fermi support|
|compute_30 and compute_32 | + Kepler support + Unified memory programming|
|compute_35 | + Dynamic parallelism support|
|compute_50, compute_52, and compute_53 | + Maxwell support|


---

## CUDA Architecture
### Equipment
* **主機 (host)**  ：顯示卡那台PC。
* **裝置 (device)**：顯示卡。
* **核心 (kernel)**：顯示卡上執行的程式碼區段。


### Hardware
* **SP(Streaming Process)**: 最基本的處理單元，一個SP可執行一個thread。 <br />
* **SM(Streaming Multiprocessor)**: 由多個SP加上一些資源而組成，每個SM所擁有之SP數量依據不同GPU架構而不同，Kepler=128,Maxwell=128,Pascal=128，任一GPU則有多個SM，軟體定義上SP是平行運算，但物理實現上並非所有SP能夠同時平行運算，有些會處於掛起、就緒等其他狀態。 <br />


### Software
* **thread**: 一個CUDA的併行程序(kernel)會以多個threads來執行，為3D結構。 <br />
* **block**: 由多個threads所組成，同一block中之threads可以同步，也可利用shared memory來共享資料，為3D結構。 <br />
* **grid**: 由多個blocks所組成，只能為2D結構。 <br />
* **warp**: 32個threads組成一個warp，warp是調度和運行的基本單元。warp中所有threads並行的執行相同的指令，一個warp需要佔用一個SM運行，多個warps需要輪流進入SM，故只能有一個warp正被執行。 <br />
  * **active warp**: 指已分配給SM的warp，且該warp所需資源如暫存器也已分配。 <br />
  * **resident thread**: 為一個正在SM裡同時執行之warp數，一個GPU上resident thread最多只(SM)x32個。 <br />


### Memory
* **Registers**: 暫存器 <br />
* **Shared Memory**: 共享記憶體 <br />
* **Host Memory**: 主機記憶體 <br />
* **Device Memory**: 裝置記憶體 <br />

| name       |   position   | read/write speed |
| ---------  | ------------ | ------------ |
|  Registers |    GPU   |     immediately    |  
|  Shared Memory  |   GPU      |   4 cycles   |
|  Host Memory    |   PC Board | (PCI-E) Slow |  
|  Device Memory  |     GPU    |400-600 cycles|  


### CUDA API
* **cudaMalloc()**: 配置device記憶體 <br />
* **cudaFree()**: 釋放device記憶體 <br />
* **cudaMemcpy()**: 記憶體複製 <br />
* **cudaGetErrorstring()**: 錯誤字串解釋 <br />
* **~~cudaThreadSynchronize() 舊的~~** / **cudaDeviceSynchronize()**: 同步化，在運行內核時須用到，內核中所有線程式不同步的，直到所有任務計算完畢時須同步。 <br />
* **cudaStreamSynchronize()**: 與cudaDeviceSynchronize()類似功能，但函式帶有一參數Stream。<br />


* **kernel**: <br />
   **宣告**: <br />
      ___ global ___ void kernel_name(type1 arg1, type2 arg2, ...){ <br />
                 函式內容 <br />
      }; <br />
   **呼叫**: <br />
      kernel<<< blocks, threads >>>( arg1, arg2 ); <br />
      
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


