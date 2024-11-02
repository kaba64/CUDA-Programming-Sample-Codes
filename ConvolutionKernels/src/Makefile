CC := nvcc
TARGET_EXEC := myprogram
CFLAGS  := -arch=sm_86 -O3 -std=c++20 -diag-suppress=611 -DDEBUG 
CFLAGSCPU := -Wall -O3 -Wno-deprecated-enum-enum-conversion
SRC_DIR := .
OBJ_DIR := obj

INCLUDE = /usr/include/opencv4
LIBS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
EXECUTABLE = myprogram

$(EXECUTABLE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -Xcompiler="$(CFLAGSCPU)" $^ -o $@ $(LIBS)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -Xcompiler="$(CFLAGSCPU)" -c $< -o $@ -I$(INCLUDE)

clean:
	rm -r $(OBJ_DIR) $(EXECUTABLE) *~

run:
	@./$(EXECUTABLE)
