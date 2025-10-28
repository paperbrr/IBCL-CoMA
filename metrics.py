import numpy as np
from tabulate import tabulate

import numpy as np
from tabulate import tabulate

def calculate_metrics(performance_dict):
    num_tasks = len(performance_dict)
    metrics = {
        'Avg Final Acc': 0,
        'Avg Acc': 0,
        'Avg Forget': 0
    }
    
    
    # Final Accuracy
    final_acc = [performance_dict[i][-1] for i in range(num_tasks)]
    metrics['Avg Final Acc'] = np.mean(final_acc)
    
    # All Accuracy
    all_acc = [performance_dict[i][i] for i in range(num_tasks)]
    metrics['Avg Acc'] = np.mean(all_acc)
    
    # Forgetting
    forget_values = []
    for j in range(num_tasks - 1):  
        max_perf = max(performance_dict[j][j:]) 
        final_perf = performance_dict[j][-1]
        forget_values.append(max_perf - final_perf)
    metrics['Avg Forget'] = np.mean(forget_values) if forget_values else 0
    
    return metrics

def analyze_configurations(configurations):
    table = []
    headers = ["α", "Method", "Avg Final Acc", "Avg Acc", "Avg Forget"]
    
    for alpha, method, data in configurations:
        metrics = calculate_metrics(data)
        row = [
            alpha,
            method,
            f"{metrics['Avg Final Acc']:.4f}",
            f"{metrics['Avg Acc']:.4f}",
            f"{metrics['Avg Forget']:.4f}"
        ]
        table.append(row)
    
    return tabulate(table, headers=headers, tablefmt="grid", floatfmt=".4f")