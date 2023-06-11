# 安装方法
1. 在Makefile中设置正确的hacl路径，rt路径（前两行）
2. 设置和NPU型号有关的宏

    | NPU型号 | 宏   |
    | ------- | ---- |
    | 昇腾910A | -DASCEND910A     |
    | 昇腾910B |  -DASCEND910B    |
    | 昇腾710 |  -DASCEND710    |
3. 安装
    ```shell
    cd hablas
    mkdir lib
    mkdir build
    make install
    ```

# 使用方法
1. 包含头文件: ```#include"hablas.h"```
2. 编译时选择动态库: ```-lhablas```
3. 运行时需将动态库路径添加至系统环境变量

# todo
1. 将所有alpha，beta入参改为host指针模式
2. kernel注册时进行判断，如果已经注册过则不需要注册
3. 检查代码中的pipe_all语句，尽可能替换掉
4. API中不能含有流同步语句
5. 检查所有API是否能够被正常调用
