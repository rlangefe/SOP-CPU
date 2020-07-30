CC = nvcc -O3 -arch=compute_70 -lineinfo

EFILE = ./sop.x
OBJS = ./sop.o ./utils.o ./random_generator.o ./global.o ./energy.o ./io.o ./params.o ./neighbor_list.o ./cell_list.o ./pair_list.o ./GPUvars.o

sop.x: $(OBJS)
	@echo "linking ..."
	$(CC) -o $(EFILE) $(OBJS)

sop.o: ./sop.h ./utils.h ./random_generator.h ./global.h ./energy.h ./io.h ./params.h ./neighbor_list.h ./cell_list.o ./pair_list.h ./GPUvars.h
	$(CC) -c ./sop.cu -o ./sop.o

utils.o: ./utils.h
	$(CC) -c ./utils.cu -o ./utils.o

random_generator.o: ./random_generator.h
	$(CC) -c ./random_generator.cu -o ./random_generator.o

global.o: ./global.h ./random_generator.h
		$(CC) -c ./global.cu -o ./global.o

GPUvars.o: ./GPUvars.h ./global.h ./utils.h
		$(CC) -c ./GPUvars.cu -o ./GPUvars.o

energy.o: ./utils.h ./global.h ./energy.h ./GPUvars.h
		$(CC) -c ./energy.cu -o ./energy.o

io.o: ./global.h ./io.h
		$(CC) -c ./io.cu -o ./io.o

params.o: ./global.h ./params.h
		$(CC) -c ./params.cu -o ./params.o

neighbor_list.o: ./global.h ./neighbor_list.h ./GPUvars.h
		$(CC) -c ./neighbor_list.cu -o ./neighbor_list.o

cell_list.o: ./global.h ./cell_list.h
		$(CC) -c ./cell_list.cu -o ./cell_list.o

pair_list.o: ./global.h ./pair_list.h ./GPUvars.h ./neighbor_list.h 
		$(CC) -c ./pair_list.cu -o ./pair_list.o

clean:
	rm -f $(OBJS) $(EFILE)
