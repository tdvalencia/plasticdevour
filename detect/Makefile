CC = g++

CFLAGS = -O3 -g -Wall -Wextra
CFLAGS += -Ilib/opencv/include -Isrc/gpio -Ilib/Config -Ilib/MotorDriver -Ilib/PCA9685

LDFLAGS = -L./lib/opencv/lib -lopencv_core -lopencv_imgproc -lopencv_dnn -lopencv_videoio -lopencv_highgui
LDFLAGS += -ldl -lpthread -lwiringPi -lm

CPPFILES  = $(wildcard src/*.cpp)
CFILES	  = $(wildcard lib/**/*.c) $(wildcard src/**/*.c)

OBJ  = $(CPPFILES:.cpp=.o) $(CFILES:.c=.o)
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

%.o: %.c
	$(CC) -o $@ -c $< -D USE_WIRINGPI_LIB $(CFLAGS) -fpermissive $(LDFLAGS)

clean:
	rm -rf $(BIN) $(OBJ)
