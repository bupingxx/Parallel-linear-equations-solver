num_proc = 2
num_threads = 2
n = 2048
ToPrint = 0

build:
	mpiCC -g -Wall -o gauss1 gauss1.cpp
	g++ -g -Wall -fopenmp -o gauss2 gauss2.cpp
	mpiCC -g -Wall -fopenmp -o gauss3 gauss3.cpp

test1:
	mpiexec -n $(num_proc) ./gauss1 $(n) $(ToPrint)

test2:
	./gauss2 $(n) $(num_threads) $(ToPrint)

test3:
	mpiexec -n $(num_proc) ./gauss3 $(n) $(num_threads) $(ToPrint)


clean:
	rm gauss1, gauss2, gauss3