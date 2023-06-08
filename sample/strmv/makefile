CC := g++ 
INC := -I../../include -I/home/t_user/zsx-home/code/include -I/usr/local/Ascend/ascend-toolkit/latest/acllib/include/
LIB := -L../../lib -L/usr/local/Ascend/ascend-toolkit/latest/lib64
FLAG := -lhablas -lruntime -lerror_manager -lascendcl

strmv : strmv.cpp
	${CC} ${LIB} ${INC} -o strmv ./strmv.cpp ${FLAG} 