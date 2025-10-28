from tasks import get_split_cifar10_tasks, get_split_cifar100_tasks, get_split_svhn_tasks
from models import Model
from coma import train_continual_coma
from metrics import calculate_metrics
import torch

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    task_count = 5
    epochs = 300

    tasks = get_split_cifar100_tasks(batch_size=128, task_count=task_count, shuffle=True)
    model = Model().to(device)

    results = train_continual_coma(
        tasks=tasks,
        task_count=task_count,
        epochs=epochs,
        alpha=0.7,
        device=device
    )

    print("\nTask-wise Accuracies:")
    for task_id, accs in results.items():
        print(f"Task {task_id}: {accs}")

    metrics = calculate_metrics(results)
    print("\nPerformance Metrics")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")