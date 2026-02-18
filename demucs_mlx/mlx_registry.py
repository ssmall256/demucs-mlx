"""
Central MLX model registry (no MLX imports).
"""

MLX_MODEL_REGISTRY = {
    'htdemucs': {
        'signatures': ['955717e8'],
        'is_bag': False,
        'description': 'Standard 4-source HTDemucs',
    },
    'htdemucs_ft': {
        'signatures': ['f7e0c4bc', 'd12395a8', '92cfc3b6', '04573f0d'],
        'is_bag': True,
        'weights': [6, 1, 1, 1],
        'description': 'Fine-tuned ensemble (4 models)',
    },
    'htdemucs_6s': {
        'signatures': ['5c90dfd2'],
        'is_bag': False,
        'description': '6-source HTDemucs (drums, bass, other, vocals, guitar, piano)',
    },
    'hdemucs_mmi': {
        'signatures': ['75fc33f5'],
        'is_bag': True,
        'description': 'Hybrid Demucs v3 (MusDB + 800 songs, single-model bag)',
    },
    'mdx': {
        'signatures': ['0d19c1c6', '7ecf8ec1', 'c511e2ab', '7d865c68'],
        'is_bag': True,
        'description': 'MDX Track A bag (MusDB HQ only)',
    },
    'mdx_extra': {
        'signatures': ['e51eebcc', 'a1d90b5c', '5d2d6c55', 'cfa93e08'],
        'is_bag': True,
        'description': 'MDX Track B bag (extra data)',
    },
    'mdx_q': {
        'signatures': ['6b9c2ca1', 'b72baf4e', '42e558d4', '305bc58f'],
        'is_bag': True,
        'description': 'MDX Track A bag (DiffQ quantized)',
    },
    'mdx_extra_q': {
        'signatures': ['83fc094f', '464b36d7', '14fc6a69', '7fd6ef75'],
        'is_bag': True,
        'description': 'MDX Track B bag (DiffQ quantized)',
    },
}
