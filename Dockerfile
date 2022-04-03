FROM ubuntu:22.04
WORKDIR /workdir
COPY . .

# Define variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf-8
ENV QT_QPA_PLATFORM=offscreen
ENV TZ=US/Pacific

# Use Python 3 as the default version of Python
RUN ln --symbolic /usr/bin/python3 /usr/bin/python

# Instala modulos con pip
# mutmut is pinned to version 2.1.0 because v2.2.0 does not work with ShellSpec
RUN pip install \
    "mutmut==2.1.0" \
    black \
    flake8 \
    ipython \
    pylint \
    pyright \
    pytest \
    rope
