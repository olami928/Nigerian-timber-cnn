# Reduced CPU training profile

`cpu.yaml` is a reduced-epoch profile for local CPU execution. It retains the
same two architectures, approved specimen-grouped folds, class order, image
grid, patch size, normalization, augmentations, loss, batch size, and learning
rates as `default.yaml`. It reduces stage 1 from 10 epochs to 3, caps stage 2
at 3 epochs, and uses patience 2.

This changes the training schedule. Treat its outputs as exploratory and keep
them separate from the full Phase 2 GPU run; do not report them as if they were
trained under the approved full schedule.

From the project root, run:

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 ./.venv/bin/python -u -m src.train \
  --config configs/cpu.yaml \
  --output-dir artifacts/p2_training_cpu \
  --num-workers 2 \
  --cache-patches \
  --patch-cache-path /tmp/nigerian_timber_cpu_patch_cache.npy
```

The full GPU run continues to use `configs/default.yaml` and
`artifacts/p2_training/`. The CPU profile saves its fold and final outputs to
`artifacts/p2_training_cpu/`.
