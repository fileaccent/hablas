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
    git clone https://github.com/deltaguo/hablas.git
    cd hablas
    mkdir lib
    mkdir build
    make install
    ```

# 使用方法
1. 包含头文件: ```#include"hablas.h"```
2. 编译时选择动态库: ```-lhablas```
3. 运行时需将动态库路径添加至系统环境变量
