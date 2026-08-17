#!/bin/sh
set -eu

upgrade=0
if [ "${1:-}" = "--upgrade" ]; then
    upgrade=1
    shift
fi

apt-get update
apt-get install -y --no-install-recommends "$@"
if [ "$upgrade" -eq 1 ]; then
    apt-get upgrade -y
fi
