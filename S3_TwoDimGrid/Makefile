# Makefile for Lab 0 helloworld
#
# Compiler: NVIDIA CUDA NVCC Version 5.0
#
NVCC    		= nvcc
NVCC_FLAGS  	= -arch=sm_50

all:	message TwoDimGrid

message:
	@clear
	@echo MakeFile is executing ...

TwoDimGrid:	
	$(NVCC) $(NVCC_FLAGS) TwoDimGrid.cu -o TwoDimGrid.exe
	@echo Completed Makefile.

clean:
	rm -rf *.o *.exe
