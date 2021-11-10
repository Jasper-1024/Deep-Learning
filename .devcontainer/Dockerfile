FROM tensorflow/tensorflow:2.6.1-gpu
LABEL maintainer="JS"

# 无需等待用户输入
ENV DEBIAN_FRONTEND noninteractive

ENV USERNAME="user"
ENV USER_HOME="/home/${USERNAME}"

# 创建用户
RUN apt-get update &&  \
    apt-get install -y sudo tree && \
    adduser --disabled-password --gecos '' ${USERNAME} && \
    adduser ${USERNAME} sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers 
# 安装依赖库
RUN apt-get update && apt-get install -y \
    vim wget git \
    python3-dev python3-pip && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR ${USER_HOME}
USER "${USERNAME}"

# WORKDIR ${USER_HOME}

# RUN pip3 install numpy pandas sklearn matplotlib seaborn pyyaml h5py && \
#     pip3 install keras --no-deps && \
#     pip3 install opencv-python && \
#     pip3 install imutils && \
#     mkdir notebooks


# Store notebooks in this mounted directory
# VOLUME /notebooks

# USER "${USERNAME}"
