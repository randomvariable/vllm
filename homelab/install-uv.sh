#!/bin/sh
set -eu

install_dir=${UV_INSTALL_DIR:-/usr/local/bin}
tmp_dir=$(mktemp -d)
trap 'rm -rf "$tmp_dir"' EXIT

curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$tmp_dir" sh
install -m 0755 "$tmp_dir/uv" "$install_dir/uv"
install -m 0755 "$tmp_dir/uvx" "$install_dir/uvx"
