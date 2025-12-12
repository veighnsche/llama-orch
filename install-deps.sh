#!/bin/bash
# Install dependencies on Fedora for rbee build

echo "Installing Node.js..."
sudo dnf install -y nodejs npm

echo "Installing pnpm using npm..."
sudo npm install -g pnpm

echo "Installing Rust..."
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

echo "Installing wasm-pack..."
cargo install wasm-pack

echo "Installing GTK and GLib development libraries..."
sudo dnf install -y glib2-devel gtk3-devel pkg-config

echo "Installing additional build tools..."
sudo dnf install -y gcc gcc-c++ make openssl-devel

echo "All dependencies installed!"
