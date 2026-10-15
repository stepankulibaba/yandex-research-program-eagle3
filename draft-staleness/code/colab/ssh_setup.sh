#!/bin/bash
# `ssh colab-eagle CMD` runs commands on the Colab session "eagle". Colab allows ONE ssh connection at a time.
set -e
mkdir -p ~/.ssh && chmod 700 ~/.ssh
[ -f ~/.ssh/id_ed25519 ] || ssh-keygen -q -t ed25519 -N "" -f ~/.ssh/id_ed25519 -C colab-cli
grep -q "Host colab-eagle" ~/.ssh/config 2>/dev/null || cat >> ~/.ssh/config <<CFG
Host colab-eagle
  User root
  ProxyCommand $HOME/.local/bin/colab ssh --proxy-mode -s eagle
  StrictHostKeyChecking no
  UserKnownHostsFile /dev/null
  LogLevel ERROR
  ServerAliveInterval 30
CFG
chmod 600 ~/.ssh/config
