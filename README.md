# QRS Independent P Wave Augmentation for ECG Delineation

Version 1.0.0 accompanies the manuscript *QRS-Independent P-Wave Augmentation and Multi-Source Training for Cross-Database ECG Delineation in Pathological Rhythms* by Ali Akdağlı, Hüseyin Yanık, Ali Yıldız and Ahmet Çelik. The package supports reconstruction of the reported analysis from archived event counts.

## Reproduce the analysis

With Python 3.12 available, run the following from the repository root in an isolated Python environment.

```sh
python -m pip install -r requirements.txt
python analysis/replay.py --output replay_output
```

The command regenerates and checks the primary BUT PDB comparisons and pooled metrics, the four European ST-T contrasts with participant-clustered intervals and Holm adjustment, and the 16 descriptive rhythm subgroup rows in Supplementary Table S13. It reads the inputs in `results/` and writes the regenerated outputs to the chosen output directory. The reference run used Python 3.12.10, NumPy 2.5.3 and SciPy 1.18.1.

## Files

- `analysis/` contains the standalone replay entry point and the original BUT PDB score and resampling code.
- `results/but_pdb/` contains the frozen source-record counts, archived pooled metrics and primary contrasts, rhythm tags, and timing results.
- `results/external/` contains the European ST-T record counts, external contrasts, and supplementary source-record sensitivity analyses.
- `results/manuscript/` contains the tables supporting selected manuscript figures, timing and rhythm results, and the literature comparison.
- `training_source/` preserves the model, augmentation, evaluation and training source associated with original study commit `a829ded0cb6884c7e0dfb41770b25b99ab9d487a`. The optional training workflow uses its own dependencies and requires source datasets arranged as described in [the training notes](training_source/README.md).

## Attribution

Use [CITATION.cff](CITATION.cff) when citing this software version. The MIT license applies to this code and associated documentation. Read [Data sources and attribution](DATA_SOURCES.md) for the provenance and source terms of the analysis inputs.
