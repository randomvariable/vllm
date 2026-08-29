#!/bin/sh
set -eu

install_dir=${UV_INSTALL_DIR:-/usr/local/bin}
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

# INSTALLER_NO_MODIFY_PATH keeps the installer from appending a `. $tmp_dir/env`
# line to /root/.profile: the temp dir is gone by the time a container starts,
# so every `bash -lc` would open its log with a spurious "No such file" error.
curl -LsSf https://astral.sh/uv/install.sh |
  UV_INSTALL_DIR="$tmp_dir" INSTALLER_NO_MODIFY_PATH=1 sh
install -m 0755 "$tmp_dir/uv" "$install_dir/uv"
install -m 0755 "$tmp_dir/uvx" "$install_dir/uvx"
