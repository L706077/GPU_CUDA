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
|Architecture|   Pascal     |	   Pascal    |	 Pascal 	| Pascal   |
|Threads/Warp|	   32       |       32     |  	 32     |   32     |
|    TDP     |     250W	    |     180W	   |     250 W	|   75 W   |



## CUDA Architecture
### hardware
SP(Streaming Process): 最基本的處理單元，一個SP可執行一個thread <br />
SM(Streaming Multiprocessor): 由多個SP加上一些資源而組成，每個SM所擁有之SP數量依據不同GPU架構而不同，Kepler=128,Maxwell=128,Pascal=128，任一GPU則有多個SM，軟體定義上SP是平行運算，但物理實現上並非所有SP能夠同時平行運算，有些會處於掛起、就緒等其他狀態。 <br />


### software
thread: 一個CUDA的併行程序(kernel)會以多個threads來執行，為3D結構。 <br />
block: 由多個threads所組成，同一block中之threads可以同步，也可利用shared memory來共享資料，為3D結構 <br />
grid: 由多個blocks所組成，只能為2D結構 <br />
warp: 32個threads組成一個warp，warp是調度和運行的基本單元。warp中所有threads並行的執行相同的指令，一個warp需要佔用一個SM運行，多個warps需要輪流進入SM，一個GPU上resident thread最多只有SM*warp個 <br />


### Memory
Registers:  <br />
Shared Memory: <br />
Local Memory: <br />
Global Memory: <br />


