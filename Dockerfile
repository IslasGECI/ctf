FROM ubuntu:22.04
WORKDIR /workdir
COPY . .

# Define variables de entorno
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/lib/shellspec:/workdir/src:$PATH"
ENV PYTHONIOENCODING=utf-8
ENV QT_QPA_PLATFORM=offscreen
ENV TZ=US/Pacific

# Instala paquetes en el sistema operativo
RUN apt update && apt full-upgrade --yes && apt install --yes \
    curl \
    exa \
    git \
    neofetch \
    neovim \
    pip \
    ripgrep \
    tmux

# Use Python 3 as the default version of Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Instala modulos con pip
# mutmut is pinned to version 2.1.0 because v2.2.0 does not work with ShellSpec
RUN pip install \
    "mutmut==2.1.0" \
    black \
    flake8 \
    ipython \
    powerline-shell \
    pylint \
    pyright \
    pytest \
    rope

# Instala ShellSpec
RUN curl \
    --fail \
    --location https://git.io/shellspec \
    --show-error \
    --silent \
    | sh -s -- --yes
RUN shellspec --init

# Importa archivos de configuración
RUN mkdir --parents ${HOME}/repos && \
    git clone --bare https://github.com/devarops/dotfiles.git ${HOME}/repos/dotfiles.git && \
    git --git-dir=${HOME}/repos/dotfiles.git --work-tree=${HOME} checkout && \
    git --git-dir=${HOME}/repos/dotfiles.git --work-tree=${HOME} config --local status.showUntrackedFiles no

