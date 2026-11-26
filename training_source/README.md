# Training source

This directory preserves selected model definitions, training and evaluation scripts, frozen configurations, and analysis plans from the original study snapshot. The count-based reproduction at the repository root provides the portable verification path.

The training scripts derive the project root from their own location and expect source recordings under `data/raw/` within this directory. The PhysioNet paths are defined in `src/paths.py`. Additional dataset locations and labels are defined in `src/extra_data.py`.

- [LUDB](https://physionet.org/content/ludb/1.0.1/)
- [BUT PDB](https://physionet.org/content/but-pdb/1.0.0/)
- [MIT-BIH P-wave annotations](https://physionet.org/content/pwave/1.0.0/)
- [QT Database](https://physionet.org/content/qtdb/1.0.0/)
- [Zhejiang signals](https://doi.org/10.1038/s41597-020-0440-8) and [Zhejiang labels](https://doi.org/10.6084/m9.figshare.14035196.v1)
- [Resting ECG Segmentation Database](https://doi.org/10.6084/m9.figshare.28892186.v4)
- [ISP data](https://doi.org/10.5281/zenodo.14679837) and [Multi-Expert annotations](https://doi.org/10.5281/zenodo.24021589)

Training and end-to-end waveform inference require the data acquisitions and computational environment described in the study. Archived analysis outputs are supplied under `results/` at the repository root.
