CC := /home/t_user/zsx-home/code/compiler/bin/haclc
rt_path := /home/t_user/zsx-home/code/include

LIB :=
INC := -I${rt_path} -I./include
FLAG :=
ARCH := -DASCEND910B
TBE :=#-DTBE

all : install

obj := ./build/*.o


install: ./build/elf_hablas_hgemm_kernel.o ./build/elf_hablas_hgemm_batched_kernel.o ./build/elf_hablas_hgemm_strided_batched_kernel.o ./build/elf_hablas_hsyrk_kernel.o ./build/elf_hablas_hsyr2k_kernel.o ./build/elf_hablas_hgemv_kernel.o ./build/elf_hablas_sgemv_kernel.o ./build/elf_hablas_cgemv_kernel.o ./build/elf_hablas_ssymv_kernel.o ./build/elf_hablas_hsymv_kernel.o ./build/elf_hablas_csymv_kernel.o ./build/elf_hablas_htrmv_main_kernel.o ./build/elf_hablas_htrmv_copy_kernel.o ./build/elf_hablas_strmv_main_kernel.o ./build/elf_hablas_strmv_copy_kernel.o ./build/elf_hablas_ctrmv_main_kernel.o ./build/elf_hablas_ctrmv_copy_kernel.o ./build/elf_hablas_htrmm_kernel.o ./src/handle.cc ./src/hablas.cc
	g++ -fpic ${INC} -c ./src/handle.cc -o ./build/handle.o
	g++ -fpic ${INC} ${ARCH} -c ./src/hablas.cc -o ./build/hablas.o
	g++ -shared ./build/*.o -o ./lib/libhablas.so

./build/elf_hablas_hgemm_kernel.o: ./src/kernel/hgemm.cc
	${CC} -c ./src/kernel/hgemm.cc --hacl-device-only ${INC} ${TBE} -o ./build/hablas_hgemm_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_kernel.o ./build/elf_hablas_hgemm_kernel.o
	rm -f ./build/hablas_hgemm_kernel.o

./build/elf_hablas_hgemm_batched_kernel.o: ./src/kernel/hgemm_batched.cc
	${CC} -c ./src/kernel/hgemm_batched.cc --hacl-device-only ${INC} -o ./build/hablas_hgemm_batched_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_batched_kernel.o ./build/elf_hablas_hgemm_batched_kernel.o
	rm -f ./build/hablas_hgemm_batched_kernel.o

./build/elf_hablas_hgemm_strided_batched_kernel.o: ./src/kernel/hgemm_strided_batched.cc
	${CC} -c ./src/kernel/hgemm_strided_batched.cc --hacl-device-only ${INC} -o ./build/hablas_hgemm_strided_batched_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemm_strided_batched_kernel.o ./build/elf_hablas_hgemm_strided_batched_kernel.o
	rm -f ./build/hablas_hgemm_strided_batched_kernel.o

./build/elf_hablas_hsyrk_kernel.o: ./src/kernel/hsyrk.cc
	${CC} -c ./src/kernel/hsyrk.cc --hacl-device-only ${INC} -o ./build/hablas_hsyrk_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsyrk_kernel.o ./build/elf_hablas_hsyrk_kernel.o
	rm -f ./build/hablas_hsyrk_kernel.o

./build/elf_hablas_hsyr2k_kernel.o: ./src/kernel/hsyr2k.cc
	${CC} -c ./src/kernel/hsyr2k.cc --hacl-device-only ${INC} -o ./build/hablas_hsyr2k_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsyr2k_kernel.o ./build/elf_hablas_hsyr2k_kernel.o
	rm -f ./build/hablas_hsyr2k_kernel.o

./build/elf_hablas_hgemv_kernel.o: ./src/kernel/hgemv.cc
	${CC} -c ./src/kernel/hgemv.cc --hacl-device-only ${INC} -o ./build/hablas_hgemv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hgemv_kernel.o ./build/elf_hablas_hgemv_kernel.o
	rm -f ./build/hablas_hgemv_kernel.o

./build/elf_hablas_sgemv_kernel.o: ./src/kernel/sgemv.cc
	${CC} -c ./src/kernel/sgemv.cc --hacl-device-only ${INC} -o ./build/hablas_sgemv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_sgemv_kernel.o ./build/elf_hablas_sgemv_kernel.o
	rm -f ./build/hablas_sgemv_kernel.o

./build/elf_hablas_hsymv_kernel.o: ./src/kernel/hsymv.cc
	${CC} -c ./src/kernel/hsymv.cc --hacl-device-only ${INC} -o ./build/hablas_hsymv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_hsymv_kernel.o ./build/elf_hablas_hsymv_kernel.o
	rm -f ./build/hablas_hsymv_kernel.o

./build/elf_hablas_ssymv_kernel.o: ./src/kernel/sgemv.cc
	${CC} -c ./src/kernel/ssymv.cc --hacl-device-only ${INC} -o ./build/hablas_ssymv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_ssymv_kernel.o ./build/elf_hablas_ssymv_kernel.o
	rm -f ./build/hablas_ssymv_kernel.o

./build/elf_hablas_cgemv_kernel.o: ./src/kernel/cgemv.cc
	${CC} -c ./src/kernel/cgemv.cc --hacl-device-only ${INC} -o ./build/hablas_cgemv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_cgemv_kernel.o ./build/elf_hablas_cgemv_kernel.o
	rm -f ./build/hablas_cgemv_kernel.o

./build/elf_hablas_csymv_kernel.o: ./src/kernel/csymv.cc
	${CC} -c ./src/kernel/csymv.cc --hacl-device-only ${INC} -o ./build/hablas_csymv_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_csymv_kernel.o ./build/elf_hablas_csymv_kernel.o
	rm -f ./build/hablas_csymv_kernel.o

./build/elf_hablas_htrmv_main_kernel.o: ./src/kernel/htrmv_main_kernel.cc
	${CC} -c ./src/kernel/htrmv_main_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_htrmv_main_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_htrmv_main_kernel.o ./build/elf_hablas_htrmv_main_kernel.o
	rm -f ./build/hablas_htrmv_main_kernel.o
./build/elf_hablas_htrmv_copy_kernel.o: ./src/kernel/htrmv_copy_kernel.cc
	${CC} -c ./src/kernel/htrmv_copy_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_htrmv_copy_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_htrmv_copy_kernel.o ./build/elf_hablas_htrmv_copy_kernel.o
	rm -f ./build/hablas_htrmv_copy_kernel.o

./build/elf_hablas_strmv_main_kernel.o: ./src/kernel/strmv_main_kernel.cc
	${CC} -c ./src/kernel/strmv_main_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_strmv_main_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_strmv_main_kernel.o ./build/elf_hablas_strmv_main_kernel.o
	rm -f ./build/hablas_strmv_main_kernel.o
./build/elf_hablas_strmv_copy_kernel.o: ./src/kernel/strmv_copy_kernel.cc
	${CC} -c ./src/kernel/strmv_copy_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_strmv_copy_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_strmv_copy_kernel.o ./build/elf_hablas_strmv_copy_kernel.o
	rm -f ./build/hablas_strmv_copy_kernel.o

./build/elf_hablas_ctrmv_main_kernel.o: ./src/kernel/ctrmv_main_kernel.cc
	${CC} -c ./src/kernel/ctrmv_main_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_ctrmv_main_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_ctrmv_main_kernel.o ./build/elf_hablas_ctrmv_main_kernel.o
	rm -f ./build/hablas_ctrmv_main_kernel.o
./build/elf_hablas_ctrmv_copy_kernel.o: ./src/kernel/ctrmv_copy_kernel.cc
	${CC} -c ./src/kernel/ctrmv_copy_kernel.cc --hacl-device-only ${INC} -o ./build/hablas_ctrmv_copy_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_ctrmv_copy_kernel.o ./build/elf_hablas_ctrmv_copy_kernel.o
	rm -f ./build/hablas_ctrmv_copy_kernel.o

./build/elf_hablas_htrmm_kernel.o: ./src/kernel/htrmm.cc
	${CC} -c ./src/kernel/htrmm.cc --hacl-device-only ${INC} -o ./build/hablas_htrmm_kernel.o
	./bin/run_elf_change_hacl_kernel ./build/hablas_htrmm_kernel.o ./build/elf_hablas_htrmm_kernel.o
	rm -f ./build/hablas_htrmm_kernel.o

clean:
	rm ./build/elf_hablas_hgemm_batched_kernel.o
	rm ./lib/libhablas.so
