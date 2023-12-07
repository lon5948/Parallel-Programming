CXX := mpicxx
CXXFLAGS := -O3 -Wall -Wno-unused-result

all: matmul

matmul: main.cc matmul.cc
	$(CXX) $(CXXFLAGS) main.cc matmul.cc -o matmul

.PHONY: clean

clean:
	rm -rf matmul *.o *~