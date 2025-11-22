# TEAM_556: Node dev CLIs from upstream prebuilt binaries (pnpm, wrangler, biome) for rbee devShells
final: prev:
let
  system = final.stdenv.hostPlatform.system;
in
{
  wrangler =
    if system == "x86_64-linux" then
      final.stdenvNoCC.mkDerivation {
        pname = "wrangler";
        version = "1.21.0";
        src = final.fetchurl {
          url = "https://github.com/cloudflare/wrangler-legacy/releases/download/v1.21.0/wrangler-v1.21.0-x86_64-unknown-linux-musl.tar.gz";
          sha256 = "sha256-IJhFX2JDXpUijPxFBXVfpxivtfpyR2Ju9Y6lL71vn9g=";
        };
        dontUnpack = true;
        dontBuild = true;
        installPhase = ''
          mkdir -p "$out/bin"
          tar -xzf "$src"
          cp dist/wrangler "$out/bin/wrangler"
          chmod +x "$out/bin/wrangler"
        '';
      }
    else
      prev.wrangler;

  biome =
    if system == "x86_64-linux" then
      final.stdenvNoCC.mkDerivation {
        pname = "biome";
        version = "2.3.7";
        src = final.fetchurl {
          url = "https://github.com/biomejs/biome/releases/download/@biomejs/biome%402.3.7/biome-linux-x64-musl";
          sha256 = "sha256-AcouCOVEZhOlnEq1zvufbATQ9J6LQgauq80LsAwAzwc=";
        };
        dontUnpack = true;
        dontBuild = true;
        installPhase = ''
          mkdir -p "$out/bin"
          cp "$src" "$out/bin/biome"
          chmod +x "$out/bin/biome"
        '';
      }
    else
      prev.biome;

  pnpm =
    if system == "x86_64-linux" then
      final.stdenvNoCC.mkDerivation {
        pname = "pnpm";
        version = "10.23.0";
        src = final.fetchurl {
          url = "https://github.com/pnpm/pnpm/releases/download/v10.23.0/pnpm-linuxstatic-x64";
          sha256 = "sha256-v1htnuf0xFM6Bn3SNopSEl+mAbyoWEnL/77gqUNsuRw=";
        };
        dontUnpack = true;
        dontBuild = true;
        installPhase = ''
          mkdir -p "$out/bin"
          cp "$src" "$out/bin/pnpm"
          chmod +x "$out/bin/pnpm"
        '';
      }
    else
      prev.pnpm;
}
