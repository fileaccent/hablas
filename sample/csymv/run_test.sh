uplo=$1
M=$2
lda=$3
incx=$4
incy=$5

#uplo 0 or 1

# eval "export PATH=/usr/local/Ascend/ascend-toolkit/5.1/compiler/ccec_compiler/bin:$PATH"
# eval "export USE_LOW_LEVEL=1"
eval "make"
eval "python3 data_gen_csymv.py $uplo $M $lda $incx $incy"
eval "./csymv $uplo $M $lda $incx $incy"