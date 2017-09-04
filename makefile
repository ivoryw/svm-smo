LFLAGS += -larmadillo

CXXFLAGS += -Wall -Wextra -Wshadow -Wnon-virtual-dtor -Wpedantic -g 
CPPFLAGS += -std=c++14
OBJS =  svm.o

all: svm
svm: main.o $(OBJS)
	$(CXX) $^ -o $@ $(LFLAGS) 

%.o: %.cc
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

PHONY: .clean
clean:
	rm *.o 

