import pickle as pk

def sort(data_containers, criterion):
    return sorted(data_containers,key=criterion)[::-1]

def top_N(data_containers, criterion, n_best):
    return sort(data_containers, criterion)[:min(n_best, len(data_containers))]

def conditional_removal(data_containers, condition, keep=False):
    return [container for container in data_containers if not((not keep and condition(container)) or (keep and not condition(container)))]

def import_data(data_containers, import_path, pickle=True):
    if pickle:
        with open(import_path, "rb") as file:
            data_containers.extend(pk.Unpickler(file).load())
    else:
        raise NotImplementedError()
    return data_containers

def export_data(data_containers, export_path, pickle=True):
    if pickle:
        with open(export_path, "wb") as file:
            pk.Pickler(file).dump(data_containers)
    else:
        with open(export_path, "w") as file:
            file.write(str(data_containers))
    return data_containers