FROM tensorflow/tensorflow:2.2.1-gpu-py3-jupyter
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

WORKDIR ${USER_HOME}

RUN pip3 install numpy pandas sklearn matplotlib seaborn pyyaml h5py && \
    pip3 install keras --no-deps && \
    pip3 install opencv-python && \
    pip3 install imutils && \
    mkdir notebooks

# RUN ["mkdir", "notebooks"]

COPY conf/.jupyter ${USER_HOME}/.jupyter
COPY run_jupyter.sh /

# Jupyter and Tensorboard ports
EXPOSE 8888 6006

# Store notebooks in this mounted directory
# VOLUME /notebooks

USER "${USERNAME}"

# RUN RUN apt-get update && \
#     apt-get install -y sudo && \
#     # 创建用户目录 / 不将用户登录添加到数据库 /指定shell
#     useradd -m -l -s /bin/bash user 

# USER user
# WORKDIR /home/user

# CMD ["/run_jupyter.sh"]