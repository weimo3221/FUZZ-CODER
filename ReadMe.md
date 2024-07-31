## 步骤1：安装软件环境

- 利用conda搭建3.9环境(或者有3.9环境的python也可以)

```
$ conda create -n work python=3.9
$ conda activate work
```
- 安装相关库(注意这里使用的cuda的版本，cuda的版本不一定要统一，但是其他python库的版本必须统一)

```
$ pip install torch==1.12.0+cu113 torchaudio==0.12.0+cu113 torchvision==0.13.0+cu113 pandas tqdm einops timm flask 
```
- 安装afl-cov

~~~
$ apt-get install lcov
$ unzip afl-cov-master.zip
然后就可以看到一个afl-cov-master的文件夹
~~~

- 配置最基础的afl-fuzz，这一步可以做可以不做，主要是为了方便其他afl工具的使用，比如afl-gcc等

~~~
$ tar -xf afl-latest.tgz
将用于llm测试的afl-fuzz.c代码复制到AFL中
$ cp afl-fuzz-time-llm.c ./afl-2.52b
$ cd afl-2.52b
$ sudo make
$ sudo make install
~~~

## 步骤2：在8个程序中测试某个大模型

**请务必按照在每次测试前按照以下步骤进行测试**

### 步骤2.1：配置搭载llm的服务器

~~~
进入./CtoPython/PyDOC

1.测试的内容修改module_app_model.py中第8行的内容：
program-n = {程序名}（程序名的内容包含：nm、objdump、readelf、libxml、mp3gain、magick、tiffsplit、libjpeg，一共8个程序）
比如：program-n = "nm"

2.测试的内容修改module_app_model.py中第16行的内容：
base_model = {大模型的文件夹所在位置}
比如：base_model = "/data2/hugo/fuzz/code_llama/model-checkpoint/model-checkpoint200/dpsk7b/cpfs01/shared/public/yj411294/fuzzy_test/stanford_alpaca/models/dpsk-7B/lr3e-5-ws100-wd0.0-bsz512/checkpoint-200"

3.有多个卡进行测试，需要修改module_app_model.py和module_client.py中的端口号，避免导致冲突。
保证module_app_model.py第106行中port={端口号}和module_client.py第15行url = 'http://127.0.0.1:{端口号}/'相同

4.新建一个session运行llm的服务器
$ screen -S {session名字}（注意不要重复）
进入./CtoPython/PyDOC
$ python module_app_model.py
$ ctrl+a+d退出当前session（下一次进入该session通过screen -r {session名字}）
~~~

### 步骤2.2：安装对应程序（一共8个）

~~~
根据步骤2.1中第1步选择的程序进行安装，8个程序的安装命令如下：
****注意，即使已经安装过需要删除重新安装****

1.objdump、nm和readelf的安装
$ tar -xf binutils-2.27.tar.gz
$ cd binutils-2.27
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下:
$ ./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将binutils安装的话，可以继续执行sudo make install，测试的话到上面一步即可)

2.xmllint的安装
$ tar -xf libxml2-2.9.2.tar.gz
$ cd libxml2-2.9.2
$ ./autogen.sh
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将libxml2安装的话，可以继续执行sudo make install，测试的话到上面一步即可)

3.mp3gain的安装
$ tar -xf mp3gain-1.5.2.tar.gz
$ cd mp3gain-1.5.2/
$ vim Makefile
修改其中CCS的值，把gcc改为$afl-gcc$，$afl-gcc$是afl编译器的路径
CC=$afl-gcc$ -fprofile-arcs -ftest-coverage
比如举个例子:
CC=/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage
$ make
(如果你想将mp3gain安装的话，可以继续执行sudo make install，测试的话到上面一步即可)

4.magick的安装
$ tar xvfz ImageMagick-7.1.0-49.tar.gz
$ cd ImageMagick-7.1.0-49
$ sudo apt-get install build-essential libjpeg-dev libpng-dev libtiff-dev libgif-dev zlib1g-dev libfreetype6-dev libfontconfig1-dev
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make
(如果你想将ImageMagick安装的话，可以继续执行sudo make install，测试的话到上面一步即可)

5.tiffsplit的安装
$ tar -xf libtiff-Release-v3-9-7.tar.gz
$ cd libtiff-Release-v3-9-7
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make
(如果你想将libtiff安装的话，可以继续执行sudo make install，测试的话到上面一步即可)

6.jpegtran的安装
$ tar -xf jpegsrc.v9e.tar.gz
$ cd jpeg-9e
其中$afl-gcc$和$afl-g++$是这两个编译器的路径，这两个编译器可以在afl-2.52b文件夹中找到
$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
比如举个例子，可以写成如下
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make
(如果你想将libjpeg安装的话，可以继续执行sudo make install，测试的话到上面一步即可)
~~~

### 步骤2.3：测试程序并进行数据处理（一共8个）

~~~
根据步骤2.1和步骤2.2选择的程序进行测试和测试后的数据处理，8个程序的测试命令如下：

1.nm
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../nm_out ../binutils-2.27/binutils/nm-new -a @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../nm_out -e "../binutils-2.27/binutils/nm-new -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../nm_out -p nm
最后将nm_out文件夹打包以供数据分析和统计

2.objdump
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../objdump_out ../binutils-2.27/binutils/objdump -x -a -d @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../objdump_out -e "../binutils-2.27/binutils/objdump -x -a -d AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../objdump_out -p objdump
最后将objdump_out文件夹打包以供数据分析和统计

3.readelf
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../readelf_out ../binutils-2.27/binutils/readelf -a @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../readelf_out -e "../binutils-2.27/binutils/readelf -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../readelf_out -p readelf
最后readelf_out文件夹打包以供数据分析和统计

4.xmllint
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i ../xml_in/ -o ../xml_out ../libxml2-2.9.2/xmllint --valid --recover @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../xml_out -e "../libxml2-2.9.2/xmllint --valid --recover AFL_FILE" -c ../libxml2-2.9.2 --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../xml_out -p xmllint
最后xml_out文件夹打包以供数据分析和统计

5.mp3gain
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i ../mp3_in/ -o ../mp3_out ../mp3gain-1.5.2/mp3gain @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../mp3_out -e "../mp3gain-1.5.2/mp3gain AFL_FILE" -c ../mp3gain-1.5.2 --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../mp3_out -p mp3gain
最后mp3_out文件夹打包以供数据分析和统计

6.magick
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/images/gif -o ../gif_out ../ImageMagick-7.1.0-49/utilities/magick identify @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../gif_out -e "../ImageMagick-7.1.0-49/utilities/magick identify AFL_FILE" -c ../ImageMagick-7.1.0-49/utilities --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../gif_out -p magick
最后gif_out文件夹打包以供数据分析和统计

7.tiffsplit
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i ../tiff_in/ -o ../tiff_out ../libtiff-Release-v3-9-7/tools/tiffsplit @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../tiff_out -e "../libtiff-Release-v3-9-7/tools/tiffsplit AFL_FILE" -c ../libtiff-Release-v3-9-7/tools/ --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../tiff_out -p tiffsplit
最后tiff_out文件夹打包以供数据分析和统计

8.jpegtran
$ screen -S {session名字}（注意不要重复）
$ cd afl-2.52b
$ ./afl-fuzz -i ../jpg_in/ -o ../jpg_out ../jpeg-9e/jpegtran @@
待Fuzz完毕后进入afl-cov-master文件夹，执行下面命令
$ ./afl-cov -d ../jpg_out -e "../jpeg-9e/jpegtran AFL_FILE" -c ../jpeg-9e --enable-branch-coverage --overwrite
待上面命令执行完毕后进入afl-2.52b，执行下面命令
$ python afl-showmap.py -f ../jpg_out -p jpegtran
最后jpg_out文件夹打包以供数据分析和统计
~~~

### 步骤3：记录测试过程中的必须步骤

ssh连接：ssh zhongyuan_peng@36.137.226.47 -p 11130

密码：casia123

执行服务器命令：sudo /data/zhongyuan_peng/anaconda3/envs/fuzz/bin/python module_app_model.py

sudo /data/zhongyuan_peng/anaconda3/envs/fuzz/bin/python module_app.py

scp连接：

scp -P 11130 zhongyuan_peng@36.137.226.47:/data2/ghc/fuzz/test_env/gif-dpsk7b/test.tar.gz D:\school-works\paper-fighting\download-exchange

tar打包： tar -czvf 123.tar.gz 123 456

查看进程并杀死： ps -ef | grep 进程名字；kill -9 PID(第2个数字)

针对afl-cov不能用的问题：[afl覆盖率统计工具afl-cov常见问题总结-CSDN博客](https://blog.csdn.net/happygogf/article/details/106433043)

scp -P 11130 zhongyuan_peng@36.137.226.47:/data2/ghc/fuzz/result.tar.gz D:\school-works\paper-fighting\重启项目\实验结果\dpsk7b