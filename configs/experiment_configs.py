# Base parameters shared across configurations unless overridden
BASE_TEMPLATE = {
    'vocab_size': 512,
    'block_size': 128,
    'bias': False,
    'dropout': 0.1,
    'ffn_multiplier': 4,
    'learning_rate': 6e-4,
    'batch_size': 32,
    'max_iters': 15000, #1000
    'eval_interval': 100,
    'eval_iters': 20
}

CONFIGS = {
    # --- Task 4.1: Baseline Configuration ---
    'task_4_1_baseline': {
        **BASE_TEMPLATE,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
    },
    
    # --- Task 4.2: Hyperparameter Sensitivity Sweeps ---
    'task_4_2_high_lr': {
        **BASE_TEMPLATE,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'learning_rate': 5e-3, # Aggressive learning rate adjustment
    },
    'task_4_2_large_context': {
        **BASE_TEMPLATE,
        'n_layer': 4,
        'n_head': 4,
        'n_embd': 128,
        'block_size': 256, # Doubled sequence capacity window
    },

    # --- Task 4.3: Structural Scaling Experiments ---
    'task_4_3_tiny': {
        **BASE_TEMPLATE,
        'n_layer': 2,
        'n_head': 2,
        'n_embd': 64,
    },
    'task_4_3_medium': {
        **BASE_TEMPLATE,
        'n_layer': 6,
        'n_head': 6,
        'n_embd': 192,
    },
    'task_4_3_large': {
        **BASE_TEMPLATE,
        'n_layer': 8,
        'n_head': 8,
        'n_embd': 256,
    },
    
    # --- Part 3: Architecture Ablation Experiments ---
    'task_3_2_no_pos': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'use_positional_embeddings': False # Disables positional encoding
    },
    'task_3_3_no_mask': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'use_causal_masking': False # Allows looking into the future
    },
    'task_3_4_single_head': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 1, 'n_embd': 128, # Single head bottleneck
    },
    'task_3_5_small_ffn': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'ffn_multiplier': 1 # Aggressively reduces FFN capacity from 4x to 1x
    },
    'task_3_6_no_residual': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'bypass_residuals': True # Breaks gradient flow
    },
    'task_3_6_no_ln': {
        **BASE_TEMPLATE, 'n_layer': 4, 'n_head': 4, 'n_embd': 128,
        'bypass_ln': True # Causes exploding gradients
    }
}
