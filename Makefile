# export USE_LOW_LEVEL=1
CC := /home/t_user/zsx-home/code/compiler/bin/haclc
rt_path := /home/t_user/zsx-home/code/include

LIB :=
INC := -I${rt_path} -I./include
FLAG :=
ARCH := -DASCEND910B
TBE :=#-DTBE

KERNEL_DIR := ./src/kernel
BUILD_DIR := ./build

SRCS := $(wildcard $(KERNEL_DIR)/*.cc)
OBJS := $(patsubst $(KERNEL_DIR)/%.cc, $(BUILD_DIR)/elf_hablas_%_kernel.o, $(SRCS))

all: install

install: $(OBJS) ./src/handle.cc ./src/hablas.cc
	g++ -fpic $(INC) -c ./src/handle.cc -o $(BUILD_DIR)/handle.o
	g++ -fpic $(INC) $(ARCH) -c ./src/hablas.cc -o $(BUILD_DIR)/hablas.o
	rm -f $(BUILD_DIR)/hablas_*
	g++ -shared $(BUILD_DIR)/*.o -o ./lib/libhablas.so

$(BUILD_DIR)/elf_hablas_%.o: $(BUILD_DIR)/hablas_%.o
	./bin/run_elf_change_hacl_kernel $< $@

$(BUILD_DIR)/hablas_%_kernel.o: $(KERNEL_DIR)/%.cc
	$(CC) -c $< --hacl-device-only $(INC) $(TBE) -o $@

clean:
	rm -f $(BUILD_DIR)/*.o ./lib/libhablas.so
