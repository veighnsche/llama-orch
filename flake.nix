{
  description = "rbee monorepo dev + build surfaces (Node + Rust)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-25.05";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      rust-overlay,
    }:
    let
      lib = nixpkgs.lib;

      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = lib.genAttrs systems;

      pkgsFor = system:
        import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            rust-overlay.outputs.overlays.default
          ];
        };

      mkRustToolchain = pkgs:
        let
          rust-bin = rust-overlay.outputs.lib.mkRustBin { } pkgs;
        in
        rust-bin.stable.latest.default.override {
          extensions = [
            "rust-analyzer"
            "rust-src"
            "clippy-preview"
            "rustfmt-preview"
          ];
        };

      pnpmCorepackHook = ''
        source "${./nix/pnpm-corepack-hook.sh}"
      '';

      rustEnvHook = ''
        export CARGO_HOME="$PWD/.cargo"
        export RUSTUP_HOME="$PWD/.rustup"
        export CARGO_TARGET_DIR="$PWD/target"
      '';

      fullStackHook = ''
        ${rustEnvHook}
        ${pnpmCorepackHook}
      '';
    in
    {
      formatter = forAllSystems (system: (pkgsFor system).nixfmt-rfc-style);

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          rustToolchain = mkRustToolchain pkgs;
        in
        {
          # TEAM_526: cargo xtask wrapper so CI / dev shells can call via `nix run .#xtask -- <cmd>`
          default = pkgs.writeShellApplication {
            name = "rbee-xtask";
            runtimeInputs = [ rustToolchain pkgs.pkg-config ];
            text = ''
              export CARGO_HOME="${toString "${self}/.cargo"}"
              exec cargo xtask "$@"
            '';
          };
        }
      );

      apps = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          mkPnpmApp = name: script:
            {
              type = "app";
              program = pkgs.writeShellScript name ''
                ${pnpmCorepackHook}
                exec pnpm ${script} "$@"
              '';
            };
        in
        {
          dev-all = mkPnpmApp "rbee-dev-all" "run dev:all";
          dev-ui = mkPnpmApp "rbee-dev-ui" "run dev:ui";
          build-ui = mkPnpmApp "rbee-build-ui" "run build:ui";
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          rustToolchain = mkRustToolchain pkgs;

          commonNodeTooling = with pkgs; [
            biome
            nodejs_22
          ];
          rustCliTools = with pkgs; [
            rustup
            rustToolchain
            cargo-edit
            cargo-watch
            cargo-outdated
            cargo-expand
            wasm-pack
            wasm-bindgen-cli
            binaryen
          ];
          nativeInputs = with pkgs; [
            pkg-config
            cmake
            ninja
            python312
            wrapGAppsHook4
          ];
          commonBuildInputs = with pkgs; [
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
            libsoup_3
            webkitgtk_4_1
            libdrm
            libxkbcommon
            libGL
            libGL.dev
            openssl
          ];
          commonHook = fullStackHook;
        in
        {
          default = pkgs.mkShell {
            packages = commonNodeTooling ++ rustCliTools;
            nativeBuildInputs = nativeInputs;
            buildInputs = commonBuildInputs;
            shellHook = ''
              ${commonHook}
              echo "[rbee] full-stack shell → $(rustc --version) | node $(node --version)"
            '';
          };

          frontend = pkgs.mkShell {
            packages = commonNodeTooling;
            nativeBuildInputs = [ pkgs.wrapGAppsHook4 ];
            buildInputs = commonBuildInputs;
            shellHook = ''
              ${pnpmCorepackHook}
              echo "[rbee] frontend shell → node $(node --version)"
            '';
          };

          backend = pkgs.mkShell {
            packages = rustCliTools;
            nativeBuildInputs = nativeInputs;
            buildInputs = commonBuildInputs;
            shellHook = ''
              ${rustEnvHook}
              echo "[rbee] backend shell → $(rustc --version)"
            '';
          };
        }
      );
    };
}
