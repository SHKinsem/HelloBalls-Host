# cd ./yolo11_demo/cpp/build
# cmake ..
# make

cd /home/sunrise/Documents/HelloBalls-Host/yolo11_demo/cpp
rm -rf build
mkdir -p build && cd build
cmake ..
make -j8

cd /home/sunrise/Documents/HelloBalls-Host/yolo11_demo/python && python test_yolo11_api.py