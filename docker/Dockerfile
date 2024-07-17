FROM ubuntu:22.04

USER root

ENV DEBIAN_FRONTEND=noninteractive \
    TIMEZONE=Asia/Taipei

# Setting TimeZone
RUN apt -yq update && \
    apt -yq install tzdata && \
    ln -fs /usr/share/zoneinfo/${TIMEZONE} /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata

# Install Python and OpenCV
RUN apt -yq install python3-pip && \
    apt -yq install libgl1-mesa-dev libglib2.0-0 libsm6 libxrender1 libxext6

# Install requirement package
COPY [ "./requirements.txt", "/tmp/requirements.txt" ]
RUN pip install --upgrade pip && \
    pip install -r /tmp/requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu && \
    rm /tmp/requirements.txt

# Move to workspace
WORKDIR /workspace

# Entry
CMD [ "bin/bash" ]