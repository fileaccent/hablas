import sys
from sys import argv
import numpy as np

def dump_data(input_data, name, fmt, data_type):
    if fmt == "binary" or fmt == "bin":
        f_output = open(name, "wb")
        if (data_type == "float32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float32(elem.real).tobytes())
                f_output.write(np.float32(elem.imag).tobytes())
        

def calc_expect_func(n, type, incx, incy, lda):
    #A
    input1_i = np.random.uniform(0, 1.0, size=(lda, n)).astype(np.float32)
    input1_r = np.random.uniform(0, 1.0, size=(lda, n)).astype(np.float32)
    input1 = 1j * input1_i + input1_r
    #X
    input2_i = np.random.uniform(0, 1.0, size=(n * incx, 1)).astype(np.float32)
    input2_r = np.random.uniform(0, 1.0, size=(n * incx, 1)).astype(np.float32)
    input2 = 1j * input2_i + input2_r
    #Y
    input3_i = np.random.uniform(0, 1.0, size=(n * incy, 1)).astype(np.float32)
    input3_r = np.random.uniform(0, 1.0, size=(n * incy, 1)).astype(np.float32)
    input3 = 1j * input3_i + input3_r

    # print(input1)
    # print(input2)
    # print(input3)

    # for i in range(0, n) :
    #     for j in range(i + 1, n) :
    #         input1_r[i][j] = 0
    # print(input1_r)

    dump_data(input1.ravel('F'), "./data/input1.bin", fmt = "binary", data_type = "float32")
    dump_data(input2.ravel('F'), "./data/input2.bin", fmt = "binary", data_type = "float32")
    dump_data(input3.ravel('F'), "./data/input3.bin", fmt = "binary", data_type = "float32")

    # print(input1)

    alpha_i = np.random.uniform(0, 1, size=(1, 1)).astype(np.float32)
    alpha_r = np.random.uniform(0, 1, size=(1, 1)).astype(np.float32)
    alpha = 1j * alpha_i + alpha_r
    # print(alpha)
    dump_data(alpha, "./data/alpha.bin", fmt = "binary", data_type = "float32")

    beta_i = np.random.uniform(0, 1, size=(1, 1)).astype(np.float32)
    beta_r = np.random.uniform(0, 1, size=(1, 1)).astype(np.float32)
    beta = 1j * beta_i + beta_r
    # print(beta)
    dump_data(beta, "./data/beta.bin", fmt = "binary", data_type = "float32")

    real_input1 = np.random.uniform(-1.0, 1.0, size=(n, n)).astype(complex)

    # print(input1)
    for i in range(n):
        for j in range(n):
            real_input1[i][j] = input1[i][j]
    # print(real_input1)
    # print(input2)
    real_input2 = input2[0::incx]
    
    real_input3 = input3[0::incy]
    print(real_input3[n - 1])

    if type == 0:
        for i in range(n):
            for j in range(i + 1, n):
                real_input1[i][j] = real_input1[j][i]
    else:
        for j in range(n):
            for i in range(j + 1, n):
                real_input1[i][j] = real_input1[j][i]
    y = np.matmul(real_input1, real_input2) * alpha + real_input3 * beta 
    for i in range(n):
        input3[i * incy] = y[i]
    # print(input3)
    dump_data(input3.ravel('F'), "./data/expect.bin", fmt = "binary", data_type = "float32")



if __name__ == "__main__":
    mode = int(argv[1])
    n = int(argv[2])
    lda = int(argv[3])
    incx = int(argv[4])
    incy = int(argv[5])
    calc_expect_func(n, mode, incx, incy, lda)