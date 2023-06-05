CC := /home/t_user/gyj/build/bin/haclc
rt_path := /home/t_user/gyj/test/include

LIB :=
INC := -I${rt_path} -I./include
FLAG :=
ARCH := -DASCEND910B
TBE :=#-DTBE
all: hsymv


install: hgemm hgemm_batched hgemm_strided_batched hsyrk hsyr2k hgemv sgemv hsymv ./src/handle.cc ./src/hablas.cc
	g++ -fpic ${INC} -c ./src/handle.cc -o ./build/handle.o
	g++ -fpic ${INC} ${ARCH} -c ./src/hablas.cc -o ./build/hablas.o
	g++ -shared ./build/handle.o ./build/hablas.o ./build/elf_hablas_hgemm_kernel.o ./build/elf_hablas_hgemm_batched_kernel.o ./build/elf_hablas_hgemm_strided_batched_kernel.o ./build/elf_hablas_hsyrk_kernel.o ./build/elf_hablas_hsyr2k_kernel.o ./build/elf_hablas_hgemv_kernel.o ./build/elf_hablas_sgemv_kernel.o ./build/elf_hablas_hsymv_kernel.o -o ./lib/libhablas.so

hgemm: ./src/kernel/hgemm.cc
	${CC} -c ./src/kernel/hgemm.cc --hacl-device-only ${INC} ${TBE} -o ./build/hablas_hgemm_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_kernel.o ./build/elf_hablas_hgemm_kernel.o

hgemm_batched: ./src/kernel/hgemm_batched.cc
	${CC} -c ./src/kernel/hgemm_batched.cc --hacl-device-only ${INC} -o ./build/hablas_hgemm_batched_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_batched_kernel.o ./build/elf_hablas_hgemm_batched_kernel.o

hgemm_strided_batched: ./src/kernel/hgemm_strided_batched.cc
	${CC} -c ./src/kernel/hgemm_strided_batched.cc --hacl-device-only ${INC} -o ./build/hablas_hgemm_strided_batched_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_strided_batched_kernel.o ./build/elf_hablas_hgemm_strided_batched_kernel.o

hsyrk: ./src/kernel/hsyrk.cc
	${CC} -c ./src/kernel/hsyrk.cc --hacl-device-only ${INC} -o ./build/hablas_hsyrk_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsyrk_kernel.o ./build/elf_hablas_hsyrk_kernel.o

hsyr2k: ./src/kernel/hsyr2k.cc
	${CC} -c ./src/kernel/hsyr2k.cc --hacl-device-only ${INC} -o ./build/hablas_hsyr2k_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsyr2k_kernel.o ./build/elf_hablas_hsyr2k_kernel.o

hgemv: ./src/kernel/hgemv.cc
	${CC} -c ./src/kernel/hgemv.cc --hacl-device-only ${INC} -o ./build/hablas_hgemv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemv_kernel.o ./build/elf_hablas_hgemv_kernel.o

sgemv: ./src/kernel/sgemv.cc
	${CC} -c ./src/kernel/sgemv.cc --hacl-device-only ${INC} -o ./build/hablas_sgemv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_sgemv_kernel.o ./build/elf_hablas_sgemv_kernel.o

hsymv: ./src/kernel/hsymv.cc
	${CC} -c ./src/kernel/hsymv.cc --hacl-device-only ${INC} -o ./build/hablas_hsymv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsymv_kernel.o ./build/elf_hablas_hsymv_kernel.o

clean:
	rm ./build/*