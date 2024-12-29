<H1 align="center">
Object Tracking with YOLOv9 and DeepSORT on Jetson Nano</H1>


## I. Device Information

- **Main Device**: Jetson Nano Developer Kit eMMC
- **CPU**: Quad-core ARM A57 @ 1.43 GHz
- **GPU**: 128-core Maxwell
- **RAM**: 4 GB 64-bit LPDDR4 25.6 GB/s
- **Storage**: eMMC (16GB onboard storage) + 64GB SD card
- **Camera**: Logitech HD 720p
- **Operating System**: Ubuntu 18.04 LTS (JetPack SDK)
- **PyTorch Version**: v1.8.0
- **Python Version**: 3.6.9
- **OpenCV**: 4.5.1 with CUDA

## II. Steps to Set Up Jetson Nano
- Create SwapFile
```
sudo fallocate -l 4G /var/swapfile 
sudo chmod 600 /var/swapfile
sudo mkswap /var/swapfile
sudo swapon /var/swapfile
sudo bash -c 'echo "/var/swapfile swap swap defaults 0 0"  >> /etc/fstab' 
```
- Install these Dependencies before installing OpenCV:
```
sudo sh -c "echo '/usr/local/cuda/lib64' >> /etc/ld.so.conf.d/nvidia-tegra.conf";
sudo ldconfig;
sudo apt-get install build-essential cmake git unzip pkg-config;
sudo apt-get install libjpeg-dev libpng-dev libtiff-dev;
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev;
sudo apt-get install libgtk2.0-dev libcanberra-gtk*;
sudo apt-get install python3-dev python3-numpy python3-pip;
sudo apt-get install libxvidcore-dev libx264-dev libgtk-3-dev;
sudo apt-get install libtbb2 libtbb-dev libdc1394-22-dev;
sudo apt-get install libv4l-dev v4l-utils;
sudo apt-get install libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev;
sudo apt-get install libavresample-dev libvorbis-dev libxine2-dev;
sudo apt-get install libfaac-dev libmp3lame-dev libtheora-dev;
sudo apt-get install libopencore-amrnb-dev libopencore-amrwb-dev;
sudo apt-get install libopenblas-dev libatlas-base-dev libblas-dev;
sudo apt-get install liblapack-dev libeigen3-dev gfortran;
sudo apt-get install libhdf5-dev protobuf-compiler;
sudo apt-get install libprotobuf-dev libgoogle-glog-dev libgflags-dev;
pip3 isntall --upgrade pip;
pip3 install --upgrade setuptools;
```
- Install OpenCV 4.5
```
cd ~
wget -O opencv.zip https://github.com/opencv/opencv/archive/4.5.1.zip 
wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.5.1.zip 
unzip opencv.zip 
unzip opencv_contrib.zip

# Now rename the directories. Type each command below, one after the other.
mv opencv-4.5.1 opencv
mv opencv_contrib-4.5.1 opencv_contrib
rm opencv.zip
rm opencv_contrib.zip

# build OpenCV:
cd ~/opencv
mkdir build
cd build
cmake -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules -D EIGEN_INCLUDE_PATH=/usr/include/eigen3 -D WITH_OPENCL=OFF -D WITH_CUDA=ON -D CUDA_ARCH_BIN=5.3 -D CUDA_ARCH_PTX="" -D WITH_CUDNN=ON -D WITH_CUBLAS=ON -D ENABLE_FAST_MATH=ON -D CUDA_FAST_MATH=ON -D OPENCV_DNN_CUDA=ON -D ENABLE_NEON=ON -D WITH_QT=OFF -D WITH_OPENMP=ON -D WITH_OPENGL=ON -D BUILD_TIFF=ON -D WITH_FFMPEG=ON -D WITH_GSTREAMER=ON -D WITH_TBB=ON -D BUILD_TBB=ON -D BUILD_TESTS=OFF -D WITH_EIGEN=ON -D WITH_V4L=ON -D WITH_LIBV4L=ON -D OPENCV_ENABLE_NONFREE=ON -D INSTALL_C_EXAMPLES=OFF -D INSTALL_PYTHON_EXAMPLES=OFF -D BUILD_NEW_PYTHON_SUPPORT=ON -D BUILD_opencv_python3=TRUE -D OPENCV_GENERATE_PKGCONFIG=ON -D BUILD_EXAMPLES=OFF ..
make -j4 #This command below will take a long time (around 2 hours)
```
- Finish the install:
```
cd ~
sudo rm -r /usr/include/opencv4/opencv2
cd ~/opencv/build
sudo make install
sudo ldconfig
make clean
sudo apt-get update
```
- Verify OpenCV Installation
```
#open python3 shell
python3
import cv2
cv2._version_
```
- Install Pytorch for Jetson: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
```
# torch v1.8.0
wget https://nvidia.box.com/shared/static/p57jwntv436lfrd78inwl7iml6p13fzh.whl -O torch-1.8.0-cp36-cp36m-linux_aarch64.whl
sudo apt-get install python3-pip libopenblas-base libopenmpi-dev libomp-dev
pip3 install 'Cython<3'
pip3 install numpy torch-1.8.0-cp36-cp36m-linux_aarch64.whl
```
```
# torchvision v0.9.0
sudo apt-get install libjpeg-dev zlib1g-dev libpython3-dev libopenblas-dev libavcodec-dev libavformat-dev libswscale-dev
git clone --branch v0.9.0 https://github.com/pytorch/vision torchvision
cd torchvision
export BUILD_VERSION=0.9.0  
python3 setup.py install --user
cd ../
pip install 'pillow<7'
```

- Install albumentations to keep opencv version
```commandline
pip install albumentations --no-deps
```

- Monitoring software for Jetson Nano
```
cd ~
sudo -H pip3 install -U jetson-stats 
sudo reboot
jtop
```

## III. Steps to run Code

- Clone the repository
```
git clone https://github.com/levanhoailinh/object-tracking-withy-yolov9-and-deepSORT.git
```
- Goto the cloned folder.
```
cd object-tracking-withy-yolov9-and-deepSORT
```
- Install requirements with mentioned command below.
```
pip install -r requirements.txt
```
- Download the pre-trained YOLOv9 model weights
[yolov9](https://github.com/WongKinYiu/yolov9/releases/download/v0.1/yolov9-c.pt)


- Run code
```
#for detection and tracking
python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0

#for WebCam
python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source 0 --device 0

#for External Camera
python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source 1 --device 0

#For LiveStream (Ip Stream URL Format i.e "rtsp://username:pass@ipaddress:portno/video/video.amp")
python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source "your IP Camera Stream URL" --device 0

#for specific class (person)
python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --classes 0

#for detection and tracking with trails 
!python3 detect_dual_tracking.py --weights 'yolov9-c.pt' --source 'your video.mp4' --device 0 --draw-trails 
```

- Output file will be created in the ```working-dir/runs/detect/obj-tracking``` with original filename

### IV. Results
![alt text](PersonTracking.png)
