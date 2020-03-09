import os
import time
import torch
from collections import OrderedDict


def rename_model_weights_dict_keys(weights_path, key_map, save=False, module_only=False):
    weights_dict = torch.load(weights_path, map_location=torch.device("cpu"))
    model_dict = weights_dict['model']

    prefix = 'module.'
    if not all(key.startswith(prefix) for key in model_dict.keys()):
        prefix = ''

    changed = False
    stripped_state_dict = OrderedDict()
    for key, value in model_dict.items():
        renamed = False
        for old_prefix, new_prefix in key_map.items():
            old_prefix = prefix + old_prefix
            new_prefix = prefix + new_prefix
            if key.startswith(old_prefix):
                stripped_state_dict[key.replace(old_prefix, new_prefix)] = value
                print("{} -> {}".format(old_prefix, new_prefix))
                renamed = True
                changed = True
        if not renamed:
            stripped_state_dict[key] = value

    if module_only:
        if len(weights_dict.keys()) > 1: 
            weights_dict = True
        weights_dict = {}

    weights_dict['model'] = stripped_state_dict

    if save and changed:
        os.rename(weights_path, weights_path + '.bak.' + str(time.time())[-6:])
        torch.save(weights_dict, weights_path)
        print('Saving checkpoint done.')
    return weights_dict


if __name__ == '__main__':
    key_map = {
        'Anchor_Head.Output.Conf': 'Anchor_Head.head.conf',
        'Anchor_Head.Output.Loc': 'Anchor_Head.head.loc',
        'Anchor_Head.Head.extras.0': 'Conv_Body_FPN.extras.extre_1',
        'Anchor_Head.Head.extras.1': 'Conv_Body_FPN.extras.extre_2',
        'Anchor_Head.Head.extras.2': 'Conv_Body_FPN.extras.extre_3',
        'Anchor_Head.Head.extras.3': 'Conv_Body_FPN.extras.extre_4',
        'Anchor_Head.Head.extras.4': 'Conv_Body_FPN.extras.extre_5',
        'Anchor_Head.L2Norm': 'Conv_Body_FPN.l2norm'
    }
    weights_path = 'ckpts/ssd/mscoco/ssd_VGG16_300x300-PROB02_0.25x_test/model_latest.pth'
    rename_model_weights_dict_keys(weights_path, key_map, save=True)
