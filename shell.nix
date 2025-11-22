{ system ? builtins.currentSystem }:

let
  nixpkgs = builtins.getFlake "github:NixOS/nixpkgs/nixpkgs-unstable";
  rustOverlay = builtins.getFlake "github:oxalica/rust-overlay";

  pkgs = import nixpkgs {
    inherit system;
    overlays = [ rustOverlay.outputs.overlays.default ];
    config.allowUnfree = true;
  };

  rust-bin = rustOverlay.outputs.lib.mkRustBin { } pkgs;
  rustToolchain = rust-bin.selectLatestNightlyWith (
    toolchain: toolchain.default.override {
      extensions = [
        # TEAM_526: Align with niri dev shell conveniences
        "rust-analyzer"
        "rust-src"
      ];
    }
  );
in

# TEAM_526: Project-scoped dev environment for rbee (Node/Pnpm/Rust/GTK stack)
pkgs.mkShell {
  packages = with pkgs; [
    rustToolchain

    # Canonical JS toolchain
    nodejs_22
    pnpm
    turbo
    wrangler

    # Rust helper CLIs
    cargo-edit
    cargo-watch
    cargo-outdated
    cargo-expand
    wasm-pack
  ];

  nativeBuildInputs = with pkgs; [
    pkg-config
    cmake
    ninja
    python312
    wrapGAppsHook4
  ];

  buildInputs = with pkgs; [
    glib
    glib.dev
    gtk3
    gtk3.dev
    pango
    pango.dev
    harfbuzz
    harfbuzz.dev
    cairo
    cairo.dev
    gdk-pixbuf
    gdk-pixbuf.dev
  ];

  shellHook = ''
    # TEAM_526: Keep pnpm / cargo caches inside the repo to avoid polluting the system
    export CARGO_HOME="$PWD/.cargo"
    export RUSTUP_HOME="$PWD/.rustup"
    export PNPM_HOME="$PWD/.pnpm"
    export PATH="$PNPM_HOME:$PATH"

    echo "[rbee] dev shell ready → Rust $(rustc --version) | Node $(node --version) | pnpm $(pnpm --version)"
  '';
}
