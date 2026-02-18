# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""
Easier training for reproducibility
"""

from ._explorers import MyExplorer


@MyExplorer
def explorer(launcher):
    launcher.slurm_(
        gpus=8,
        time=3 * 24 * 60,
        partition='devlab,learnlab')

    launcher.bind_({'ema.epoch': [0.9, 0.95]})
    launcher.bind_({'ema.batch': [0.9995, 0.9999]})
    launcher.bind_({'epochs': 600})

    base = {'model': 'demucs', 'demucs_mlx.dconv_mode': 0, 'demucs_mlx.gelu': False,
            'demucs_mlx.lstm_layers': 2}
    newt = {'model': 'demucs', 'demucs_mlx.normalize': True}
    hdem = {'model': 'hdemucs'}
    svd = {'svd.penalty': 1e-5, 'svd': 'base2'}

    with launcher.job_array():
        for model in [base, newt, hdem]:
            sub = launcher.bind(model)
            if model is base:
                # Training the v2 Demucs on MusDB HQ
                sub(epochs=360)
                continue

            # those two will be used in the repro_mdx_a bag of models.
            sub(svd)
            sub(svd, seed=43)
            if model == newt:
                # Ablation study
                sub()
                abl = sub.bind(svd)
                abl({'ema.epoch': [], 'ema.batch': []})
                abl({'demucs_mlx.dconv_lstm': 10})
                abl({'demucs_mlx.dconv_attn': 10})
                abl({'demucs_mlx.dconv_attn': 10, 'demucs_mlx.dconv_lstm': 10, 'demucs_mlx.lstm_layers': 2})
                abl({'demucs_mlx.dconv_mode': 0})
                abl({'demucs_mlx.gelu': False})
