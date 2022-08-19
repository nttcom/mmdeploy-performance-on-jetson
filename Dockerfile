# Compute Capability Examples
# ====
# JetsonAGXOrin: 87
# JetsonAGXXavier: 72
ARG CUDA_ARCHITECTURES="72;87"
ARG OPENCV_CUDA_ARCH="7.2,8.7"

ARG CUDA=11.4
ARG ONNXRUNTIME_VERSION=1.11.1
ARG NUMPY_VERSION=1.19.5
ARG CMAKE_VERSION=3.23.1
ARG OPENCV_VERSION=4.5.5
ARG OPENCV_PYTHON_VERSION=4.5.5.64
ARG TORCH_VERSION=1.11.0
ARG MMCV_VERSION=1.5.2
ARG MATPLOTLIB_VERSION=3.3.4
ARG LLVMLITE_VERSION=0.36.0
ARG NUMBA_VERSION=0.53.1
ARG SCIKIT_IMAGE_VERSION=0.17.2
ARG PPLCV_VERSION=0.6.3


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS download-cmake
ARG CMAKE_VERSION
WORKDIR /root/space
RUN wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-aarch64.tar.gz


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-numpy
ARG NUMPY_VERSION
WORKDIR /root/wheel
RUN pip3 wheel numpy==${NUMPY_VERSION}


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS onnxruntime-build
ARG ONNXRUNTIME_VERSION
ARG CUDA_ARCHITECTURES
ENV DEBIAN_FRONTEND=noninteractive

ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates\
    g++\
    gcc\
    make\
    git\
  && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=download-cmake /root/space/cmake-*-linux-aarch64.tar.gz /tmp/
RUN tar -zxf /tmp/cmake-*-linux-aarch64.tar.gz --strip=1 -C /usr/local

COPY --from=build-numpy /root/wheel/*.whl /root
RUN pip3 install setuptools wheel &&\
    pip3 install cython &&\
    pip3 install --no-index --find-links=/root numpy

RUN git clone --recursive https://github.com/microsoft/onnxruntime.git &&\
    cd onnxruntime && git checkout v${ONNXRUNTIME_VERSION} && git submodule update --recursive

RUN cd onnxruntime\
&& /bin/bash ./build.sh\
    --config RelWithDebInfo\
    --arm64\
    --build_shared_lib\
    --use_openmp\
    --cuda_home /usr/local/cuda\
    --cudnn_home /usr/lib/aarch64-linux-gnu\
    --use_cuda\
    --use_tensorrt\
    --tensorrt_home /usr/lib/aarch64-linux-gnu\
    --skip_tests\
    --config Release\
    --build_wheel\
    --build_shared_lib\
    --update\
    --build\
    --parallel\
    --cmake_extra_defines ONNXRUNTIME_VERSION=$(cat ./VERSION_NUMBER) CMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES}

#RUN wget https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz && \
#    tar -xzvf onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}.tgz

RUN mkdir -p artifact/lib/dist && mkdir -p artifact/include &&\
    cp onnxruntime/build/Linux/Release/lib*.so* artifact/lib/ &&\
    cp onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_inline.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/providers/cpu/cpu_provider_factory.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/session/onnxruntime_session_options_config_keys.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/session/onnxruntime_run_options_config_keys.h artifact/include &&\
    cp onnxruntime/include/onnxruntime/core/framework/provider_options.h artifact/include &&\
    cp onnxruntime/VERSION_NUMBER artifact/ &&\
    cp onnxruntime/LICENSE artifact/ &&\
    cp onnxruntime/ThirdPartyNotices.txt artifact/ &&\
    cp onnxruntime/docs/Privacy.md artifact/ &&\
    cp onnxruntime/README.md artifact/ &&\
    cp onnxruntime/build/Linux/Release/dist/*.whl artifact/lib/dist/

#RUN cp -r onnxruntime-linux-aarch64-${ONNXRUNTIME_VERSION}/* artifact/

FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-opencv
ARG OPENCV_PYTHON_VERSION
WORKDIR /root/wheel
RUN apt-get update && apt-get install -y libgl1-mesa-glx && apt-get clean && rm -rf /var/lib/apt/lists/*
COPY --from=build-numpy /root/wheel/*.whl /root/wheel/
RUN pip3 install pytest-runner &&\
    pip3 install --no-index --find-links=/root/wheel numpy
RUN pip3 wheel opencv-python==${OPENCV_PYTHON_VERSION}


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-mmcv-full-wheel
ARG CUDA
ARG OPENCV_VERSION
ARG OPENCV_CUDA_ARCH
ARG MMCV_VERSION
ARG TORCH_VERSION
WORKDIR /root/wheel
RUN apt-get update && \
    apt-get install -y build-essential libavcodec-dev libavformat-dev libswscale-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev libjpeg-dev libpng-dev libtiff-dev libv4l-dev v4l-utils qv4l2 curl unzip

COPY --from=build-numpy /root/wheel/*.whl /root/wheel/
COPY --from=build-opencv /root/wheel/*.whl /root/wheel/
RUN pip3 install --no-index --find-links=/root/wheel numpy opencv-python

RUN curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -o opencv-${OPENCV_VERSION}.zip && \
    curl -L https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -o opencv_contrib-${OPENCV_VERSION}.zip && \
    unzip opencv-${OPENCV_VERSION}.zip && \
    unzip opencv_contrib-${OPENCV_VERSION}.zip && \
    cd opencv-${OPENCV_VERSION}/ && \
    mkdir release && \
    cd release/ && \
    cmake -D WITH_CUDA=ON -D WITH_CUDNN=ON -D CUDA_ARCH_BIN=${OPENCV_CUDA_ARCH} -D CUDA_ARCH_PTX="" -D OPENCV_GENERATE_PKGCONFIG=ON -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OPENCV_VERSION}/modules -D WITH_GSTREAMER=ON -D WITH_LIBV4L=ON -D BUILD_opencv_python2=ON -D BUILD_opencv_python3=ON -D BUILD_TESTS=OFF -D BUILD_PERF_TESTS=OFF -D BUILD_EXAMPLES=OFF -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_INSTALL_PREFIX=/usr/local .. && \
    make -j$(nproc) && \
    make install
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip3 wheel mmcv-full==${MMCV_VERSION} -f https://download.openmmlab.com/mmcv/dist/cu${CUDA}/torch${TORCH_VERSION}/index.html -f /root/wheel


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-matplotlib
ARG MATPLOTLIB_VERSION
RUN pip3 wheel --wheel-dir=/root/wheel matplotlib==${MATPLOTLIB_VERSION}
RUN pip3 wheel --wheel-dir=/root/wheel six==1.16.0


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-numba
ARG LLVMLITE_VERSION
ARG NUMBA_VERSION
RUN apt-get update &&\
    apt-get install -y\
    llvm-9-dev\
    libtbb2\
    libtbb-dev\
    && apt-get clean && rm -rf /var/lib/apt/lists/*
# cf. http://www.neko.ne.jp/~freewing/raspberry_pi/nvidia_jetson_nano_install_llvm_7/
# cf. https://forums.developer.nvidia.com/t/install-python-packages-librosa-jetson-tx2-developer-kit-problem/126337/4
RUN ln -s /usr/lib/llvm-9/bin/llvm-config /usr/bin/llvm-config &&\
    pip3 wheel --wheel-dir=/root/wheel llvmlite==${LLVMLITE_VERSION}
RUN mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.bak &&\
    pip3 wheel --wheel-dir=/root/wheel numba==${NUMBA_VERSION}


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-scikit-image
ARG SCIKIT_IMAGE_VERSION
RUN pip3 wheel --wheel-dir=/root/wheel scikit-image==${SCIKIT_IMAGE_VERSION}


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-mmdet
COPY space/mmdetection /root/space/mmdetection
RUN cd /root/space/mmdetection &&\
    python3 setup.py bdist_wheel
WORKDIR /root/wheel
RUN cp /root/space/mmdetection/dist/*.whl /root/wheel/


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-pplcv
ARG PPLCV_VERSION
WORKDIR /root/space
COPY --from=download-cmake /root/space/cmake-*-linux-aarch64.tar.gz /tmp/
RUN wget https://github.com/openppl-public/ppl.cv/archive/refs/tags/v${PPLCV_VERSION}.tar.gz &&\
    tar -zxvf  v${PPLCV_VERSION}.tar.gz && mv ppl.cv-${PPLCV_VERSION} ppl.cv && cd ppl.cv &&\
    export PPLCV_DIR=$(pwd) &&\
    echo -e '\n# set environment variable for ppl.cv' >> ~/.bashrc &&\
    echo "export PPLCV_DIR=$(pwd)" >> ~/.bashrc &&\
    /bin/bash ./build.sh cuda


FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3 AS build-mmdeploy
ARG CUDA_ARCHITECTURES
WORKDIR /root/space
COPY --from=download-cmake /root/space/cmake-*-linux-aarch64.tar.gz /tmp/
RUN tar -zxf /tmp/cmake-*-linux-aarch64.tar.gz --strip=1 -C /usr/local

COPY --from=onnxruntime-build /artifact/ /root/space/onnxruntime
ENV ONNXRUNTIME_DIR=/root/space/onnxruntime
RUN echo "protobuf<4.21.0" > constraints.txt &&\
    pip3 install /root/space/onnxruntime/lib/dist/*.whl -c constraints.txt

COPY --from=build-pplcv /root/space/ppl.cv/cuda-build/install/ /root/space/ppl.cv/cuda-build/install/

COPY space/mmdeploy /root/space/mmdeploy
RUN cd /root/space/mmdeploy &&\
    mkdir -p build && cd build && cmake .. \
        -DMMDEPLOY_BUILD_SDK=ON \
        -DCMAKE_CXX_COMPILER=g++ \
        -Dpplcv_DIR=/root/space/ppl.cv/cuda-build/install/lib/cmake/ppl \
        -DTENSORRT_DIR=/workspace/tensorrt \
        -DONNXRUNTIME_DIR=${ONNXRUNTIME_DIR} \
        -DMMDEPLOY_BUILD_SDK_PYTHON_API=ON \
        -DMMDEPLOY_TARGET_DEVICES="cuda;cpu" \
        -DMMDEPLOY_TARGET_BACKENDS="ort;trt" \
        -DMMDEPLOY_CODEBASES=all \
        -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCHITECTURES} &&\
    make -j$(nproc)


#######################
# Main build          #
#######################
FROM nvcr.io/nvidia/l4t-pytorch:r34.1.1-pth1.11-py3
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends\
    apt-utils\
    apt-transport-https\
    ca-certificates\
    software-properties-common\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y \
    git\
    vim\
    libsm6\
    libxext6\
    libxrender-dev\
    libgl1-mesa-glx\
    locales\
    unzip\
    libtbb2\
    libtbb-dev\
    libhdf5-dev\
    libprotobuf-dev\
    protobuf-compiler\
    python3-tk\
    python3-h5py\
    llvm-9-dev\
    libgeos-dev\
&& apt-get clean\
&& rm -rf /var/lib/apt/lists/*

RUN mkdir ~/space
RUN python3 -m pip install --upgrade pip

# cmake
COPY --from=download-cmake /root/space/cmake-*-linux-aarch64.tar.gz /tmp/
RUN tar -zxf /tmp/cmake-*-linux-aarch64.tar.gz --strip=1 -C /usr/local

# mmcv
RUN pip3 install pytest-runner
COPY --from=build-mmcv-full-wheel /root/wheel/*.whl /root/space/
COPY --from=build-numpy /root/wheel/*.whl /root/space/
RUN pip3 install --no-index --find-links=/root/space numpy opencv-python
RUN MMCV_WITH_OPS=1 FORCE_CUDA=1 pip3 install --no-index --find-links=/root/space mmcv-full

# mmdetection
COPY space/mmdetection /root/space/mmdetection
COPY --from=build-mmdet /root/wheel/*.whl /root/space/
RUN pip3 install /root/space/mmdet*.whl

# llvmlite
COPY --from=build-numba /root/wheel/*.whl /root/space/
RUN ln -s /usr/lib/llvm-9/bin/llvm-config /usr/bin/llvm-config &&\
    mv /usr/include/tbb/tbb.h /usr/include/tbb/tbb.bak &&\
    pip3 install --no-index --find-links=/root/space llvmlite numba

# onnxruntime
COPY --from=onnxruntime-build /artifact/ /root/space/onnxruntime
ENV LD_LIBRARY_PATH=/root/space/onnxruntime/lib:${LD_LIBRARY_PATH}
RUN echo "protobuf<4.21.0" > constraints.txt && \
    pip3 uninstall -y onnxruntime && \
    pip3 install /root/space/onnxruntime/lib/dist/*.whl -c constraints.txt && \
    rm constraints.txt

# mmdeploy
WORKDIR /root/space
COPY --from=build-mmdeploy /root/space/mmdeploy/ /root/space/mmdeploy/
COPY --from=build-pplcv /root/space/ppl.cv/cuda-build/install/ /root/space/ppl.cv/cuda-build/install/
RUN cd /root/space/mmdeploy/build/ && cmake --install . && cd .. && pip3 install -e .[tests] --find-links=/root/space
## apply patches
COPY space.patch /root/
RUN patch -p1 -d /root/space < /root/space.patch
## remove and later override regression test config
RUN rm -r /root/space/mmdeploy/tests/regression/

WORKDIR /root
