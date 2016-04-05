# Makefile for Lab 0 helloworld
#
# Compiler: NVIDIA CUDA NVCC Version 5.0
#
NVCC    	= nvcc
NVCC_FLAGS  	= -arch=sm_35
EXE	    	= MatrixMultiplication.exe

all:	message MatrixMultiplication

message:
	@clear
	@echo MatrixMultiplication MakeFile is executing ...

MatrixMultiplication:	
	$(NVCC) $(NVCC_FLAGS) MatrixMultiplication.cu -o $(EXE)
	@echo Completed Makefile...

clean:
	rm -rf *.o $(EXE)
