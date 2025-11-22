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
      systems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      pkgsFor = system:
        import nixpkgs {
          inherit system;
          config.allowUnfree = true;
          overlays = [
            rust-overlay.outputs.overlays.default
            (import ./nix/node-dev-tools-overlay.nix)
          ];
        };
    in
    {
      formatter = forAllSystems (system: (pkgsFor system).nixfmt-rfc-style);

      packages = forAllSystems (
        system:
        let
          pkgs = pkgsFor system;
          rust-bin = rust-overlay.outputs.lib.mkRustBin { } pkgs;
          rustToolchain = rust-bin.stable.latest.default.override {
            extensions = [
              "rust-analyzer"
              "rust-src"
              "clippy-preview"
              "rustfmt-preview"
            ];
          };
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
                export PNPM_HOME="${toString "$PWD/.pnpm"}"
                export PATH="$PNPM_HOME:$PATH"
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
          rust-bin = rust-overlay.outputs.lib.mkRustBin { } pkgs;
          rustToolchain = rust-bin.stable.latest.default.override {
            extensions = [
              "rust-analyzer"
              "rust-src"
              "clippy-preview"
              "rustfmt-preview"
            ];
          };

          commonNodeTooling = with pkgs; [
            nodejs_22
            wrangler
            biome
          ];
          rustCliTools = with pkgs; [
            rustToolchain
            cargo-edit
            cargo-watch
            cargo-outdated
            cargo-expand
            wasm-pack
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
            libdrm
            libxkbcommon
            libGL
            libGL.dev
          ];
          commonHook = ''
            export CARGO_HOME="$PWD/.cargo"
            export RUSTUP_HOME="$PWD/.rustup"
            export CARGO_TARGET_DIR="$PWD/target"
            export PNPM_HOME="$PWD/.pnpm"
            export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright"
            export PATH="$PNPM_HOME:$PATH"
          '';
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
              export PNPM_HOME="$PWD/.pnpm"
              export PLAYWRIGHT_BROWSERS_PATH="$PWD/.cache/playwright"
              export PATH="$PNPM_HOME:$PATH"
              echo "[rbee] frontend shell → node $(node --version)"
            '';
          };

          backend = pkgs.mkShell {
            packages = rustCliTools;
            nativeBuildInputs = nativeInputs;
            buildInputs = commonBuildInputs;
            shellHook = ''
              export CARGO_HOME="$PWD/.cargo"
              export RUSTUP_HOME="$PWD/.rustup"
              export CARGO_TARGET_DIR="$PWD/target"
              echo "[rbee] backend shell → $(rustc --version)"
            '';
          };
        }
      );
    };
}
