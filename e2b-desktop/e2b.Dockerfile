FROM ubuntu:22.04

RUN yes | unminimize

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
  python3-xlib \
  x11-xserver-utils \
  xfce4 \
  xfce4-goodies \
  xvfb \
  xubuntu-icon-theme \
  scrot \
  python3-pip \
  python3-tk \
  python3-dev \
  x11-utils \
  gnumeric \
  python-is-python3 \
  build-essential \
  util-linux \
  locales \
  xauth \
  gnome-screenshot \
  xserver-xorg \
  xorg

ENV PIP_DEFAULT_TIMEOUT=100 \
  PIP_DISABLE_PIP_VERSION_CHECK=1 \
  PIP_NO_CACHE_DIR=1 \
  DEBIAN_FRONTEND=noninteractive

COPY ./requirements.txt requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

COPY ./45-allow-colord.pkla /etc/polkit-1/localauthority/50-local.d/45-allow-colord.pkla

COPY ./Xauthority /home/user/.Xauthority

COPY ./start-up.sh /
RUN chmod +x /start-up.sh
