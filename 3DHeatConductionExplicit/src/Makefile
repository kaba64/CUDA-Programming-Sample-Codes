CC := nvcc
TARGET_EXEC := myprogram
CFLAGS  := -arch=sm_86 -O3 -std=c++20 -DKernel3
SRC_DIR := .
OBJ_DIR := obj

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
EXECUTABLE = myprogram

$(EXECUTABLE): $(OBJ_FILES)
	$(CC) $(CFLAGS) $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -r $(OBJ_DIR) $(EXECUTABLE) *~

run:
	@./$(EXECUTABLE)
