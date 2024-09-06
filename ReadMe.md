## STEP 1: Installation of the software environment

**Read this in other languages: [English](ReadMe.md), [中文](ReadMe_zh.md).**

- Build a Python 3.9 environment with conda (or have a Python 3.9 as well)

```
$ conda create -n work python=3.9
$ conda activate work
```
- Install the relevant libraries (note the version of cuda used here, the version of cuda does not have to be uniform, but the versions of the other python libraries must be)

```
$ pip install torch==1.12.0+cu113 torchvision==0.13.0+cu113 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu113
$ pip install pandas tqdm einops timm flask 
```
- Install afl-cov

~~~
$ apt-get install lcov
$ unzip afl-cov-master.zip
Then you can see an afl-cov-master folder()
~~~

- Install afl-fuzz

~~~
#################2.52b is installed as follows#################
$ tar -xf afl-latest.tgz
Copy afl-fuzz.c code for llm testing to AFL
$ cp afl-fuzz-time-llm.c ./afl-2.52b/afl-fuzz.c
$ cd afl-2.52b
$ sudo make
$ sudo make install
(To install afl-gcc and afl-g++)
$ cp afl-fuzz-time-llm.c afl-fuzz.c
$ sudo make
(To switching to afl which can run large language models)
#################2.57b is installed as follows#################
$ unzip afl-2.57b.zip
$ cd afl-2.57b
$ chmod 777 afl-gcc afl-showmap
$ cp afl-fuzz-ori.c afl-fuzz.c
$ sudo make
$ sudo make install
(To install afl-gcc and afl-g++)
$ cp afl-fuzz-seq2seq.c afl-fuzz.c
$ sudo make
(To switching to afl which can run large language models)
~~~

## STEP 2: Testing a large language model in 8 programs

**Please be sure to follow the steps below before each test**

### STEP 2.1: Configuring the server with llm

~~~
Go to ./CtoPython/PyDOC

1.Modify the contents of line 8 in module_app_model.py:

program-n = {program name}(The contents of the program name are included: nm、objdump、readelf、libxml、mp3gain、magick、tiffsplit、libjpeg)
For example: program-n = "nm"

2.Modify the contents of line 16 in module_app_model.py:

base_model = {Location of the folder for large language models}
For example: base_model = "/data2/hugo/fuzz/code_llama/model-checkpoint/model-checkpoint200/dpsk7b/cpfs01/shared/public/yj411294/fuzzy_test/stanford_alpaca/models/dpsk-7B/lr3e-5-ws100-wd0.0-bsz512/checkpoint-200"

3.Having multiple graphics cards to test requires modifying the port numbers in module_app_model.py and module_client.py to avoid causing conflicts:

Ensure that “port = {port number}” in module_app_model.py line 106 is the same as “url = 'http://127.0.0.1:{port number}/'” in module_client.py line 15

4.Create a new session to run llm's server
$ screen -S {session name}
Go to ./CtoPython/PyDOC
$ python module_app_model.py
$ ctrl+a+d Quit the current session
~~~

### STEP 2.2: Installation of test programs (8 in total)

~~~
Installation is performed according to the programs selected in step 1 of step 2.1, and the installation commands for the eight programs are as follows:
****Note that even if you have already installed it, you need to remove it and reinstall it.****

1.Installation of objdump, nm and readelf

$ tar -xf binutils-2.27.tar.gz
$ cd binutils-2.27

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make


2.Installation of xmllint

$ tar -xf libxml2-2.9.2.tar.gz
$ cd libxml2-2.9.2
$ ./autogen.sh

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make


3.Installation of mp3gain

$ tar -xf mp3gain-1.5.2.tar.gz
$ cd mp3gain-1.5.2/
$ vim Makefile

Change the value of CC in it, change gcc to $afl-gcc$, $afl-gcc$ is the path to the afl compiler

CC=$afl-gcc$ -fprofile-arcs -ftest-coverage
For example:
CC=/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage
$ make


4.Installation of magick

$ tar xvfz ImageMagick-7.1.0-49.tar.gz
$ cd ImageMagick-7.1.0-49
$ sudo apt-get install build-essential libjpeg-dev libpng-dev libtiff-dev libgif-dev zlib1g-dev libfreetype6-dev libfontconfig1-dev

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make


5.Installation of tiffsplit

$ tar -xf libtiff-Release-v3-9-7.tar.gz
$ cd libtiff-Release-v3-9-7

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
$ make


6.Installation of jpegtran

$ tar -xf jpegsrc.v9e.tar.gz
$ cd jpeg-9e

Where $afl-gcc$ and $afl-g++$ are the paths to these two compilers, which can be found in the afl-2.52b folder

$ ./configure --disable-shared CC="$afl-gcc$ -fprofile-arcs -ftest-coverage" CXX="$afl-g++$ -fprofile-arcs -ftest-coverage"
For example:
$ ./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage" 
$ make
~~~

### STEP 2.3: Fuzz programs and data processing (8 in total)

~~~
The fuzzy test and the data processing after the fuzzy test are performed according to the procedures selected in Steps 2.1 and 2.2, and the test commands for the eight procedures are as follows:

1.nm

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../nm_out ../binutils-2.27/binutils/nm-new -a @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../nm_out -e "../binutils-2.27/binutils/nm-new -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite


2.objdump

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../objdump_out ../binutils-2.27/binutils/objdump -x -a -d @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../objdump_out -e "../binutils-2.27/binutils/objdump -x -a -d AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite


3.readelf

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i testcases/others/elf/ -o ../readelf_out ../binutils-2.27/binutils/readelf -a @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../readelf_out -e "../binutils-2.27/binutils/readelf -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite


4.xmllint

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i ../xml_in/ -o ../xml_out ../libxml2-2.9.2/xmllint --valid --recover @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../xml_out -e "../libxml2-2.9.2/xmllint --valid --recover AFL_FILE" -c ../libxml2-2.9.2 --enable-branch-coverage --overwrite


5.mp3gain

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i ../mp3_in/ -o ../mp3_out ../mp3gain-1.5.2/mp3gain @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../mp3_out -e "../mp3gain-1.5.2/mp3gain AFL_FILE" -c ../mp3gain-1.5.2 --enable-branch-coverage --overwrite


6.magick

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -m none -i testcases/images/gif -o ../gif_out ../ImageMagick-7.1.0-49/utilities/magick identify @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../gif_out -e "../ImageMagick-7.1.0-49/utilities/magick identify AFL_FILE" -c ../ImageMagick-7.1.0-49/utilities --enable-branch-coverage --overwrite


7.tiffsplit

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i ../tiff_in/ -o ../tiff_out ../libtiff-Release-v3-9-7/tools/tiffsplit @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../tiff_out -e "../libtiff-Release-v3-9-7/tools/tiffsplit AFL_FILE" -c ../libtiff-Release-v3-9-7/tools/ --enable-branch-coverage --overwrite


8.jpegtran

$ screen -S {session name}
$ cd afl-2.52b
$ ./afl-fuzz -i ../jpg_in/ -o ../jpg_out ../jpeg-9e/jpegtran @@

When Fuzz is complete go to the afl-cov-master folder and execute the following command

$ ./afl-cov -d ../jpg_out -e "../jpeg-9e/jpegtran AFL_FILE" -c ../jpeg-9e --enable-branch-coverage --overwrite
~~~
