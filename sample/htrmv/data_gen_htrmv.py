import sys
import numpy as np

def dump_data(input_data, name, fmt, data_type):
    if fmt == "binary" or fmt == "bin":
        f_output = open(name, "wb")
        if (data_type == "float16"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float16(elem).tobytes())
        elif (data_type == "float32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float32(elem).tobytes())
        elif (data_type == "int32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int32(elem).tobytes())
        elif (data_type == "int8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.int8(elem).tobytes())
        elif (data_type == "uint8"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.uint8(elem).tobytes())
    else:
        f_output = open(name, "w")
        index = 0
        for elem in np.nditer(input_data):
            f_output.write("%f\t" % elem)
            index += 1
            if index % 512 == 0:
                f_output.write("\n")

def calc_expect_func(m, lda, incx, uplo, trans, diag):
    input1 = np.random.uniform(0, 1, size=(lda, m,)).astype(np.float16)

    input2 = np.random.uniform(0, 1, size=(m * incx,)).astype(np.float16)
    
    dump_data(input1.ravel('F'),
              "./data/input1" + ".bin",
              fmt="binary",
              data_type="float16")
    dump_data(input2.ravel('F'),
              "./data/input2" + ".bin",
              fmt="binary",
              data_type="float16")

    if (uplo == 0): 
        for i in range(0, m):
            for j in range(0, m):
                if (i < j): 
                    input1[i][j] = 0
                if (diag == 1 and j == i) :
                    input1[i][j] = 1
                
    else:
        for i in range(0, m):
            for j in range(0, m):
                if (i > j):
                    input1[i][j] = 0
                if (diag == 1 and j == i) :
                    input1[i][j] = 1

    input1_true = input1[0:m, :]
    input2_true = input2[::incx]
    if trans == 0 :
        y_ture = (np.matmul(input1_true.astype(np.float32), input2_true.astype(np.float32))).astype(np.float16) 
    else :
        y_ture = (np.matmul(input2_true.astype(np.float32), input1_true.astype(np.float32))).astype(np.float16) 

    # print(y_ture)
    y = input2
    for i in range(0, m):
        y[i * incx] = y_ture[i] 
    
    # print(y)

    

    dump_data(y.ravel('F'),
              "./data/expect" + ".bin",
              fmt="binary",
              data_type="float16")

if __name__ == "__main__":
    M = int(sys.argv[1])
    lda = int(sys.argv[2])
    incx = int(sys.argv[3])
    uplo = int(sys.argv[4])
    trans = int(sys.argv[5])
    diag = int(sys.argv[6])
    if lda < M :
        lda = M
    calc_expect_func(M, lda, incx, uplo, trans, diag)

