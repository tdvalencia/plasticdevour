CC = g++

CFLAGS = -std=c++11 -O3 -g -Wall -Wextra
CFLAGS += -Ilib/opencv/include -Isrc/gpio

LDFLAGS = -L./lib/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_videoio
LDFLAGS += -lm -ldl -lpthread -lwiringPi

SRC  = $(wildcard src/*.cpp) $(wildcard src/**/*.cpp)
OBJ  = $(SRC:.cpp=.o)
BIN = bin

.PHONY: all clean

all: dirs test

dirs:
	mkdir -p ./$(BIN)
	cp -r mob bin/

run: all
	$(BIN)/test

test: $(OBJ)
	$(CC) -o $(BIN)/test $^ $(LDFLAGS)

%.o: %.cpp
	$(CC) -o $@ -c $< $(CFLAGS)

clean:
	rm -rf $(BIN) $(OBJ)
