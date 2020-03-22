from collections import OrderedDict


def fix_state_dict(state_dict):

    assert isinstance(state_dict, OrderedDict)

    if list(state_dict.keys())[0].split('.')[0] == 'module':
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
    else:
        new_state_dict = state_dict
    return new_state_dict