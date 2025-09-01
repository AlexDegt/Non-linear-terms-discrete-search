import torch

def dump_device_info():
    '''
        Provides dump of all GPUs memory usage.
    '''
    num_device = torch.cuda.device_count()
    print(f'A number of available devices: {num_device}')
    print(f'Current device: {torch.cuda.current_device()}')
    for j_device in range(num_device):
        torch.cuda.set_device(j_device)
        device_name = torch.cuda.get_device_name(j_device)
        print(f'{j_device + 1}. {device_name}')
        print('     Allocated: ', torch.cuda.max_memory_allocated(j_device)/ (1024 * 1024), 'MiB')
        print('     Reserved: ', torch.cuda.memory_reserved(j_device)/ (1024 * 1024), 'MiB')
        print('     Max reserved: ', torch.cuda.max_memory_reserved(j_device)/ (1024 * 1024), 'MiB')