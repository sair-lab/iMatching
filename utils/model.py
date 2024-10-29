#!/usr/bin/env python3

import os

import torch
import torch.nn as nn


def rectify_savepath(path, overwrite=False):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    save_path, save_file_dup = path, 0
    while os.path.exists(save_path) and not overwrite:
        save_file_dup += 1
        save_path = path + '.%d' % save_file_dup

    return save_path


def save_model(model, path):
    model = model.module if isinstance(model, nn.DataParallel) else model

    save_path = rectify_savepath(path)

    torch.save(model.state_dict(), save_path)
    print('Saved model: %s' % save_path)


def load_model(model, path, device='cuda', strict=False):
    state_dict = torch.load(path, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)
    if not strict:
        if len(missing_keys) > 0:
            print(f'Warning: Missing key(s): {missing_keys}')
        if len(unexpected_keys) > 0:
            print(f'Warning: Unexpected key(s): {unexpected_keys}')
    print('Loaded model: %s' % path)
    return model
