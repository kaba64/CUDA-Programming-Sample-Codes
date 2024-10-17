CC := nvcc
TARGET_EXEC := myprogram
CFLAGS  := -arch=sm_86 -std=c++20 -use_fast_math -fmad=true
CFLAGSCPU := -Wall -O3 -march=native
SRC_DIR := .
OBJ_DIR := obj

LIBIN_LTBB := /usr/include/x86_64-linux-gnu
LIBD_ltbb := /usr/lib/x86_64-linux-gnu
LIB_ltbb := -ltbb

SRC_FILES := $(wildcard $(SRC_DIR)/*.cu)
OBJ_FILES = $(patsubst $(SRC_DIR)/%.cu,$(OBJ_DIR)/%.o,$(SRC_FILES))
EXECUTABLE = myprogram

$(EXECUTABLE): $(OBJ_FILES)
	$(CC) $(CFLAGS) -Xcompiler="$(CFLAGSCPU)" $^ -o $@ -L$(LIBD_ltbb) $(LIB_ltbb)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu
	@if [ ! -d $(dir $@) ]; then mkdir -p $(dir $@); fi
	$(CC) $(CFLAGS) -Xcompiler="$(CFLAGSCPU)" -c $< -o $@ -I$(LIBIN_LTBB)

clean:
	rm -r $(OBJ_DIR) $(EXECUTABLE) *~

run:
	@./$(EXECUTABLE)
