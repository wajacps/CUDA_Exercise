# Makefile for Lab 0 helloworld
#
# Compiler: NVIDIA CUDA NVCC Version 5.0
#
NVCC    	= nvcc
NVCC_FLAGS  = -arch=sm_50
EXE	    	= VectorAdd.exe

all:	message VectorAdd

message:
	@clear
	@echo MakeFile is executing ...

VectorAdd:	
	$(NVCC) $(NVCC_FLAGS) VectorAdd.cu -o $(EXE)
	@echo Completed Makefile.

clean:
	rm -rf *.o $(EXE)
