echo "Configuring and building SSD ..."

mkdir build
cd build
cmake ..
make -j
pwd
./SSDSLAM
