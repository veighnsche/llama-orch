# PKGBUILD for llm-worker-rbee (CPU variant)
# Maintainer: rbee Core Team

pkgname=llm-worker-rbee-cpu
pkgver=0.1.0
pkgrel=1
pkgdesc="LLM worker for rbee system (CPU-only)"
arch=('x86_64' 'aarch64')
url="https://github.com/user/llama-orch"
license=('GPL-3.0-or-later')
depends=('gcc')
makedepends=('rust' 'cargo')
source=("git+https://github.com/user/llama-orch.git#branch=main")
sha256sums=('SKIP')

build() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo build --release --no-default-features --features cpu
}

package() {
    cd "$srcdir/llama-orch"
    install -Dm755 "bin/30_llm_worker_rbee/target/release/llm-worker-rbee" \
        "$pkgdir/usr/local/bin/llm-worker-rbee"
}

check() {
    cd "$srcdir/llama-orch/bin/30_llm_worker_rbee"
    cargo test --release --no-default-features --features cpu
}
