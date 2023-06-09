import sys
from sys import argv
import numpy as np

def dump_data(input_data, name, fmt, data_type):
    if fmt == "binary" or fmt == "bin":
        f_output = open(name, "wb")
        if (data_type == "float32"):
            for elem in np.nditer(input_data, op_flags=["readonly"]):
                f_output.write(np.float32(elem.real).tobytes(), )
                f_output.write(np.float32(elem.imag).tobytes())
        

def calc_expect_func(m, incx, trans, mode, diag, lad):
    input1_i = np.random.uniform(1, 1, size=(lda, m)).astype(np.float32)
    input1_r = np.random.uniform(1, 1, size=(lda, m)).astype(np.float32)
    input1 = 1j * input1_i + input1_r

    input2_i = np.random.uniform(0, 1, size=(m * incx, 1)).astype(np.float32)
    input2_r = np.random.uniform(0, 1, size=(m * incx, 1)).astype(np.float32)
    input2 = 1j * input2_i + input2_r

    # print(input2)
    # print(input1.ravel('F'))
    # print(input2)

    # for i in range(0, m) :
    #     for j in range(0, m) :
    #         input1_r[i][j] = j + 1
    # print(input1_r)

    dump_data(input1.ravel('F'), "./data/input1.bin", fmt = "binary", data_type = "float32")
    dump_data(input2.ravel('F'), "./data/input2.bin", fmt = "binary", data_type = "float32")

    r_input2 = input2[0::incx]

    r_input1 = np.random.uniform(0, 1.0, size=(m, m)).astype(complex)

    for i in range(m):
        for j in range(m):
            r_input1[i][j] = input1[i][j]

    if mode == 0:
        for i in range(m):
            for j in range(i + 1, m):
                r_input1[i][j] = complex(0, 0)
    else:
        for i in range(m):
            for j in range(0, i):
                r_input1[i][j] = complex(0, 0)
    # print(input1)
    if trans > 0 and mode == 0:
        for i in range(m):
            for j in range(i + 1, m):
                
                if trans == 2:
                    r_input1[i][j] = complex(r_input1[j][i].real, r_input1[j][i].imag * float(-1))
                else:
                    r_input1[i][j] = r_input1[j][i]
                r_input1[j][i] = complex(0, 0)
    elif trans > 0 and mode == 1:
        for i in range(m):
            for j in range(0, i):
                if trans == 2:
                    r_input1[i][j] = complex(r_input1[j][i].real, r_input1[j][i].imag * float(-1))
                else:
                    r_input1[i][j] = r_input1[j][i]
                r_input1[j][i] = complex(0, 0)

    if trans == 2:
        for i in range(m):
            r_input1[i][i] = complex(r_input1[i][i].real, r_input1[i][i].imag * float(-1))

    if diag == 1:
        for i in range(m):
            r_input1[i][i] = complex(1, 0)
            

    y = np.matmul(r_input1, r_input2)

    for i in range(m):
        input2[i * incx] = y[i]
    
    # print(input2.ravel('F'))

    dump_data(input2.ravel('F'), "./data/expect.bin", fmt = "binary", data_type = "float32")



if __name__ == "__main__":
    # m = 7
    # lda = 17
    # incx = 1
    # trans = 0
    # mode = 0
    # diag = 1
    m = int(argv[1])
    lda = int(argv[2])
    incx = int(argv[3])
    trans = int(argv[4])
    mode = int(argv[5])
    diag = int(argv[6])
    calc_expect_func(m, incx, trans, mode, diag, lda)