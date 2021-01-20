import copy

def iterable(obj):
    try:
        iter(obj)
    except Exception:
        return False
    else:
        return True

def parameter_grid(dictionary, remaining_keys=None):
    """
    A generator which iterates on and returns all possible combinations of values in iterables in *dictionary*.
    """
    if remaining_keys is None:
        remaining_keys=list(dictionary.keys())
    if remaining_keys == []:
        yield dictionary
    else:
        key = remaining_keys[0]
        if type(dictionary[key]) is not str and iterable(dictionary[key]):
            for val in dictionary[key]:
                fixed_val_dict=copy.deepcopy(dictionary)
                fixed_val_dict[key]=val
                yield from parameter_grid(fixed_val_dict, remaining_keys[1:])
        else:
            yield from parameter_grid(dictionary, remaining_keys[1:])
