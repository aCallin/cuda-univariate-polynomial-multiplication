CC = nvcc
FLAGS = -O3
SRC = src/*.cu
BIN = bin
OUT = a.exe

.PHONY: all
all:
	mkdir -p $(BIN)
	$(CC) $(FLAGS) $(SRC) -o $(BIN)/$(OUT)

.PHONY: run
run: all
	$(BIN)/$(OUT)

.PHONY: clean
clean:
	rm -r $(BIN)
