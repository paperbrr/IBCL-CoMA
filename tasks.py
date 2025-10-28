import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import numpy as np

def get_split_mnist_tasks(batch_size=64, task_count=5, shuffle=True):

    transform = transforms.Compose([
        transforms.ToTensor(),  # (1, 28, 28)
    ])

    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    tasks = {}

    for task_id in range(task_count):
        class_a = 2 * task_id
        class_b = 2 * task_id + 1
        
        train_indices = np.where(
            (train_dataset.targets == class_a) | (train_dataset.targets == class_b)
        )[0]
        test_indices = np.where(
            (test_dataset.targets == class_a) | (test_dataset.targets == class_b)
        )[0]


        task_train = Subset(train_dataset, train_indices)
        task_test  = Subset(test_dataset, test_indices)

        train_loader = DataLoader(task_train, batch_size=batch_size, shuffle=shuffle)
        test_loader  = DataLoader(task_test, batch_size=batch_size, shuffle=False)

        tasks[task_id] = {
            'train': train_loader,
            'test': test_loader,
            'classes': (class_a, class_b)
        }

    return tasks


def get_fashion_mnist_tasks(batch_size=64, task_count=5, shuffle=False):
    transform = transforms.Compose([transforms.ToTensor()])
    
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

    tasks = {}
    labels_per_task = 10 // task_count

    indices_by_label_train = {i: [] for i in range(10)}
    indices_by_label_test = {i: [] for i in range(10)}
    
    for idx, (_, label) in enumerate(train_dataset):
        indices_by_label_train[label].append(idx)
    for idx, (_, label) in enumerate(test_dataset):
        indices_by_label_test[label].append(idx)

    for task_id in range(task_count):
        task_labels = list(range(task_id * labels_per_task, (task_id + 1) * labels_per_task))
        
        train_indices = [i for label in task_labels for i in indices_by_label_train[label]]
        test_indices  = [i for label in task_labels for i in indices_by_label_test[label]]

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        task_train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
        task_test_loader  = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

        tasks[task_id] = {
            'labels': task_labels,
            'train': task_train_loader,
            'test': task_test_loader
        }

    return tasks


def get_split_cifar10_tasks(batch_size=64, task_count=5, resize=True, shuffle=True):
    classes_per_task = 10 // task_count

    transform = transforms.Compose([
        transforms.Resize(224) if resize else transforms.Lambda(lambda x: x),  # optional
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    def get_class_indices(dataset):
        class_indices = {i: [] for i in range(10)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        return class_indices

    train_class_indices = get_class_indices(train_dataset)
    test_class_indices  = get_class_indices(test_dataset)

    
    tasks = {}
    for task_id in range(task_count):
        class_range = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))
        
        train_indices = []
        test_indices = []
        for c in class_range:
            train_indices += train_class_indices[c]
            test_indices  += test_class_indices[c]
        
        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

        tasks[task_id] = {
            'train': train_loader,
            'test': test_loader,
            'classes': class_range  
        }

    return tasks

def get_split_cifar100_tasks(batch_size=64, task_count=10, resize=True, shuffle=True):

    classes_per_task = 100 // task_count

    transform = transforms.Compose([
        transforms.Resize(224) if resize else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)
    test_dataset  = datasets.CIFAR100(root='./data', train=False, download=True, transform=transform)

    def get_class_indices(dataset):
        class_indices = {i: [] for i in range(100)}
        for idx, (_, label) in enumerate(dataset):
            class_indices[label].append(idx)
        return class_indices

    train_class_indices = get_class_indices(train_dataset)
    test_class_indices  = get_class_indices(test_dataset)

    tasks = {}
    for task_id in range(task_count):
        class_range = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

        train_indices = []
        test_indices = []
        for c in class_range:
            train_indices += train_class_indices[c]
            test_indices  += test_class_indices[c]

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

        tasks[task_id] = {
            'train': train_loader,
            'test': test_loader,
            'classes': class_range
        }

    return tasks

def get_split_svhn_tasks(batch_size=64, task_count=5, resize=True, shuffle=True):
    classes_per_task = 10 // task_count

    transform = transforms.Compose([
        transforms.Resize(224) if resize else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    test_dataset  = datasets.SVHN(root='./data', split='test',  download=True, transform=transform)

    def get_class_indices(dataset):
        class_indices = {i: [] for i in range(10)}
        for idx, label in enumerate(dataset.labels):
            class_indices[int(label)].append(idx)
        return class_indices

    train_class_indices = get_class_indices(train_dataset)
    test_class_indices  = get_class_indices(test_dataset)

    tasks = {}
    for task_id in range(task_count):
        class_range = list(range(task_id * classes_per_task, (task_id + 1) * classes_per_task))

        train_indices = []
        test_indices = []
        for c in class_range:
            train_indices += train_class_indices[c]
            test_indices  += test_class_indices[c]

        if shuffle:
            np.random.shuffle(train_indices)
            np.random.shuffle(test_indices)

        train_loader = DataLoader(Subset(train_dataset, train_indices), batch_size=batch_size, shuffle=True)
        test_loader  = DataLoader(Subset(test_dataset, test_indices), batch_size=batch_size, shuffle=False)

        tasks[task_id] = {
            'train': train_loader,
            'test': test_loader,
            'classes': class_range
        }

    return tasks
