NVCC 	= nvcc
FLAGS 	= -O3
PROG 	= symgs-gpu-test

symgs-gpu-test: symgs-csr-gpu.cu
	$(NVCC) $(FLAGS) $^ -o $@

.PHONY: clean

clean:
	rm -f $(PROG)