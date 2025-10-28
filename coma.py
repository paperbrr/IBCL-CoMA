import torch
import copy
from training import train, test
from models import Model

def compute_coma_model(model_old, model_new, alpha=0.7):
    print(f'alpha: {alpha}')
    state_dict_old = model_old.state_dict()
    state_dict_new = model_new.state_dict()
    averaged_state_dict = copy.deepcopy(model_new.state_dict())
    for key in state_dict_new:
        if not (key.startswith('encoder.shared') or key.startswith('encoder.mu')):
            continue
        averaged_state_dict[key] = alpha * state_dict_new[key] + (1 - alpha) * state_dict_old[key]
    model_new.load_state_dict(averaged_state_dict)

def train_continual_coma(tasks, task_count, epochs, alpha, device='cuda'):
    running_test_accs = {i: [] for i in range(task_count)}
    previous_model = Model().to(device)
    buffer = []
    unsampled_z = []

    for task_id, task in tasks.items():
        print(f'training task: {task_id}')
        train_loader = task['train']
        new_model = copy.deepcopy(previous_model)
        train(model=new_model, epochs=epochs, train_loader=train_loader, buffer=buffer, unsampled_z=unsampled_z)

        if task_id > 0:
            print('averaging')
            compute_coma_model(model_old=previous_model, model_new=new_model, alpha=alpha)
        previous_model = new_model

        for test_task_id in range(task_count):
            test_acc = 0 if test_task_id > task_id else test(tasks[test_task_id]['test'], new_model, device)
            running_test_accs[test_task_id].append(test_acc)
    return running_test_accs