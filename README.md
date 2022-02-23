### Parallel-linear-equations-solver

___

#### 一、介绍

​	实现一个并行的线性方程组求解器，编程语言为C++ ，并行编程模型为MPI 与 OpenMP。



#### 二、特点

##### 1. 串行求解方法

​	使用高斯消元法，这是线性代数规划中的一个基础的算法，主要分为两个步骤，一是矩阵的上三角化，二是消元求解。

​	对于线性方程组：

![img](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-1.png)

​	可以表示成`Ax=b`的形式：

![img](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-2.png)

​	第一步是进行上三角化，在每列中找到系数最大的那行，记录该行下标`max_index`，并标记该行为选中状态`calculated`，然后在其他行中进行削减，使得除最大行以外的所有行该列元素均为0。

​	另外，由于随机生成的矩阵不保证每行按大小顺序排列，所以这里没有进行“行交换”的操作，即结果的矩阵是伪上三角矩阵，因此，我们需要维护一个三角化的次序，以便在第二步消元时，按照从后往前的顺序消元。如何维护呢？既然这是一个先进后出的过程，我们使用STL中的`stack`来维护即可，代码如下：

```c++
    stack<int> elim;
	//  Upper triangulation
    for (int j = 0; j < n; j++) {
        double max_coff = 0;
        int max_index;
        for (int i = 0; i < n; i++) {
            if (calculated[i] == -1 && abs(A[i * n + j]) > abs(max_coff)) {
                max_coff = A[i * n + j];
                max_index = i;
            }
        }
        calculated[max_index] = j;
        elim.push(max_index);
        
        for (int i = 0; i < n; i++) {
            if (calculated[i] == -1) {
                double tmp_coff = A[i * n + j] / A[max_index * n + j];
                A[i * n + j] = 0;
                for (int k = j + 1; k < n; k++) {
                    A[i * n + k] -= tmp_coff * A[max_index * n + k];
                }
                b[i] -= tmp_coff * b[max_index];
            }
        }
    }
```

​	第二步，对生成的上三角矩阵消元，消元的顺序是栈中下标的顺序，把结果存储到数组`x`中：

```c++
    // elimination element and get a solution
    for (int j = n - 1; j >= 0; j--) {
        int cur_index = elim.top();
        elim.pop();
        calculated[cur_index] = -1;
        for (int i = 0; i < n; i++) {
            if (calculated[i] != -1) {
                double tmp_coff = A[i * n + j] / A[cur_index * n + j];
                A[i * n + j] = 0;
                b[i] -= tmp_coff * b[cur_index];
            }
        }
        b[cur_index] /= A[cur_index * n + j];
        A[cur_index * n + j] = 1;
        x[j] = b[cur_index];
    }
```

​	完成串行版本后，就可以利用并行编程模型进行并行化了。



##### 2. MPI

> ​	MPI是一个跨语言的通讯协议，用于编写并行计算机。支持点对点和广播。MPI是一个信息传递应用程序接口，包括协议和和语义说明，他们指明其如何在各种实现中发挥其特性。MPI的目标是高性能，大规模性，和可移植性。

​	初步分析一下算法可以发现，主要的开销在上三角化的过程中，时间复杂度为O(n^3^)，而消元求解的时间复杂度为O(n^2^)，所以我们优先对上三角化的过程并行化。

​	进一步分析算法，我们在每列中找到一个最大的数后，就逐行进行削减，所以可以把逐行削减的任务分配给各个进程来完成，而主进程负责找到最大的数并广播该行的下标。

​	由于每一列削减时都需要知道矩阵A的信息和最大行的信息，所以在每一列三角化前，主进程应该把矩阵A和向量b分发给所有进程，在三角后所有进程应该进行一次通信，把这一次三角化的结果聚集起来。分发通过散射函数`MPI_Scatter()`实现，聚集则通过函数`MPI_Gather()`实现，广播则调用`MPI_Bcast()`。

​	分析至此，就可以开始编程了，首先，应该让主进程（进程0）负责生成矩阵A和向量b，然后广播给所有进程：

```c++
    // generate matrix A and vector b in random
    if (my_rank == 0) {
        srand(time(NULL));
        GenMat(A, n);
        GenVec(b, n);

        MPI_Bcast(A, n * n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(b, n, MPI_DOUBLE, 0, comm);
    }
    else {
        MPI_Bcast(A, n * n, MPI_DOUBLE, 0, comm);
        MPI_Bcast(b, n, MPI_DOUBLE, 0, comm);
    }
```

​	接着，在第一步上三角化中，所有进程应该创建本地的局部副本来保存散射分发的数据信息，这里创建`myA`与`myb`用于存储这些数据。进程0在列中找到最大数后，把最大行及其对应的b广播给所有进程并散射A和b。在主进程找到最大行后，其他进程接收其广播和散射的数据，然后在本地的副本上进行削减操作，最后再把结果聚集起来。

​	第二步消元求解由于任务较少且通信很多，所以没有必要并行化，让主进程完成即可。



##### 3. OpenMP

> ​	OpenMP是专门针对共享地址空间的平行计算机提供的并行计算库，在Intel C++和Visual C++ 8.0里通过#pragma支持。用OpenMP，可以不必去写诸如CreateThread之类的线程管理代码，多线程程序写起来比较简洁。

​	根据前面的分析我们知道可并行化的部分有两处，一是上三角化过程中的内层循环，二是消元求解的内层循环，由于OpenMP的特性，我们只需要在这两个循环前加上`#prgma`指令即可。



#### 三、注意

##### 1. 环境

​	编程语言C++，操作系统为 Linux， 使用的MPI库为openmpi。



##### 2. 编译

​	每个文件的开头都包含了编译和运行的参考指令，请查看代码。



#### 四、性能对比

​	在Linux上进行测试，将结果制成表格如下：

##### 1. MPI-time

| n/proc | 1        | 2        | 4        |
| ------ | -------- | -------- | -------- |
| 256    | 0.027151 | 0.027859 | 0.053277 |
| 512    | 0.222534 | 0.307815 | 0.443524 |
| 1024   | 3.37963  | 3.773810 | 4.055260 |
| 2048   | 27.4662  | 27.38871 | 29.34071 |

##### 2. MPI-speedup

| n/proc | 1    | 2    | 4    |
| ------ | ---- | ---- | ---- |
| 256    | 1    | 0.97 | 0.51 |
| 512    | 1    | 0.72 | 0.50 |
| 1024   | 1    | 0.90 | 0.83 |
| 2048   | 1    | 1.00 | 0.94 |

​	可以看到，结果是很糟糕的，MPI版本得到了异常的加速比，原因是进程之间通信的开销实在是太大了，甚至使用更多的进程反而得到了更差的效率，当n增大到一定程度时，开销的占比也许会减小，但使用MPI并行化仍然不是一个好的选择，我们再来测试OpenMP版本的结果：

##### 3. OMP-time

| n/thread | 1        | 2        | 4        |
| -------- | -------- | -------- | -------- |
| 256      | 0.019958 | 0.011538 | 0.010244 |
| 512      | 0.153317 | 0.092403 | 0.076648 |
| 1024     | 1.305182 | 0.701348 | 0.610141 |
| 2048     | 10.33150 | 5.838381 | 4.504662 |

##### 4. OMP-speedup

| n/thread | 1    | 2    | 4    |
| -------- | ---- | ---- | ---- |
| 256      | 1    | 1.73 | 1.95 |
| 512      | 1    | 1.66 | 2.00 |
| 1024     | 1    | 1.86 | 2.14 |
| 2048     | 1    | 1.77 | 2.29 |

​	可以看到，OpenMP的结果正是我们想看到的！由于省去了通信的开销，OpenMP版本获得了良好的加速比。4线程没有达到接近4的加速比可能是因为n还是太小了，由于我们使用了动态调度，可以推测程序在n很大时可以更接近线性加速比，分别用单线程和四线程测试n = 8192，结果如下：

![image-20220122025223080](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-3.png)

​	此时加速比为2.65，由于更高阶的计算耗时实在是太久了，这里便不再测试。可以看出使用OpenMP并行高斯消元法是个不错的选择。

​	对MPI+OpenMP的版本进行相同的测试，结果仍是糟糕的，在规模和线程数较少时，和MPI版本并没有太大的区别，这里不再赘述。

##### 5. 总结

​	三个版本的并行化中，OpenMP由于没有通信的开销，获得了最好的性能，在Linux系统上，不同线程的运行时间如下：

![image-20220122152355035](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-4.png)

​	在Windows系统上进行一次相同的测试，不同线程的运行时间如下：

![image-20220122152448245](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-5.png)

​	两个系统对比如下：

![image-20220122152517497](C:\Users\82458\Desktop\github\Parallel-linear-equations-solver\img\image-6.png)

​	可以看出，在相同线程时，Linux的运行时间更少，如果不考虑测试时的误差，可以看出Linux要比Windows更加稳定和快速一些，很大一部分原因是二者的系统内核不一样，导致了Linux对CPU的使用比windows更加优秀。

