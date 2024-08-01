LLMAFL_ROOT=/root/llmafl

cd $LLMAFL_ROOT
rm -rf binutils-2.27
tar -xf binutils-2.27.tar.gz
cd binutils-2.27
./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../nm_out -e "../binutils-2.27/binutils/nm-new -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../nm_out -p nm

cd $LLMAFL_ROOT
rm -rf binutils-2.27
tar -xf binutils-2.27.tar.gz
cd binutils-2.27
./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../objdump_out -e "../binutils-2.27/binutils/objdump -x -a -d AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../objdump_out -p objdump

cd $LLMAFL_ROOT
rm -rf binutils-2.27
tar -xf binutils-2.27.tar.gz
cd binutils-2.27
./configure CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../readelf_out -e "../binutils-2.27/binutils/readelf -a AFL_FILE" -c ../binutils-2.27/binutils --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../readelf_out -p readelf

cd $LLMAFL_ROOT
rm -rf libxml2-2.9.2
tar -xf libxml2-2.9.2.tar.gz
cd libxml2-2.9.2
./autogen.sh
./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../xml_out -e "../libxml2-2.9.2/xmllint --valid --recover AFL_FILE" -c ../libxml2-2.9.2 --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../xml_out -p xmllint

cd $LLMAFL_ROOT
rm -rf mp3gain-1.5.2
tar -xf mp3gain-1.5.2.tar.gz
cd mp3gain-1.5.2
sed -i '8s/.*/CC=\/usr\/local\/bin\/afl-gcc -fprofile-arcs -ftest-coverage/' Makefile
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../mp3_out -e "../mp3gain-1.5.2/mp3gain AFL_FILE" -c ../mp3gain-1.5.2 --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../mp3_out -p mp3gain

cd $LLMAFL_ROOT
rm -rf ImageMagick-7.1.0-49
tar xvfz ImageMagick-7.1.0-49.tar.gz
cd ImageMagick-7.1.0-49
./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../gif_out -e "../ImageMagick-7.1.0-49/utilities/magick identify AFL_FILE" -c ../ImageMagick-7.1.0-49/utilities --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../gif_out -p magick

cd $LLMAFL_ROOT
rm -rf libtiff-Release-v3-9-7
tar -xf libtiff-Release-v3-9-7.tar.gz
cd libtiff-Release-v3-9-7
./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../tiff_out -e "../libtiff-Release-v3-9-7/tools/tiffsplit AFL_FILE" -c ../libtiff-Release-v3-9-7/tools/ --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../tiff_out -p tiffsplit

cd $LLMAFL_ROOT
rm -rf jpeg-9e
tar -xf jpegsrc.v9e.tar.gz
cd jpeg-9e
./configure --disable-shared CC="/usr/local/bin/afl-gcc -fprofile-arcs -ftest-coverage" CXX="/usr/local/bin/afl-g++ -fprofile-arcs -ftest-coverage"
make
cd $LLMAFL_ROOT/afl-cov-master
./afl-cov -d ../jpg_out -e "../jpeg-9e/jpegtran AFL_FILE" -c ../jpeg-9e --enable-branch-coverage --overwrite
cd $LLMAFL_ROOT/afl-2.52b
python afl-showmap.py -f ../jpg_out -p jpegtran
