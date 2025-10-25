all:
	nvcc -arch=sm_80 -O3 ${CUFILES} -o ${EXECUTABLE} 
clean:
	rm -f *~ *.exe