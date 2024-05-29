#!/bin/bash

export DISPLAY=:99

Xvfb $DISPLAY -ac -screen 0 1024x768x16 &
/usr/bin/xfce4-session
