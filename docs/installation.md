# Installation

## Requirements

- Python 3.12 or 3.13
- Linux, macOS, or WSL
- Indexed BAM files for BAM-based assays

## Install from PyPI

```bash
pip install quantnado
```

## Optional Extras

```bash
# Peak calling with LanceOTron
pip install "quantnado[lanceotron]"

# Documentation and development tooling
pip install -e ".[dev]"

# Example notebook tooling
pip install "quantnado[example]"
```

## Install from Source

```bash
git clone https://github.com/Milne-Group/QuantNado.git
cd QuantNado
pip install -e .
```

## Verify the Install

```bash
quantnado --version
quantnado --help
```

## Notes

- BAM inputs must be coordinate-sorted and indexed with `.bai`.
- If you omit `--chromsizes`, QuantNado infers chromosome sizes from BAM headers or VCF contig headers when possible.
- `quantnado[lanceotron]` adds PyTorch-based peak calling support.

## Next Steps

- [Quick Start](quick_start.md)
- [Usage Guide](basic_usage.md)
- [CLI Overview](cli/index.md)
