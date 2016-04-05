# Makefile for Lab 0 helloworld
#
# Compiler: NVIDIA CUDA NVCC Version 5.0
#
NVCC    	= nvcc
NVCC_FLAGS  = -arch=sm_50
EXE	    	= Stencil1D.exe

all:	message Stencil1D

message:
	@clear
	@echo MakeFile is executing ...

Stencil1D:	
	$(NVCC) $(NVCC_FLAGS) Stencil1D.cu -o $(EXE)
	@echo Completed Makefile.

clean:
	rm -rf *.o $(EXE)
