import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils import make_dataloader_from_buffer, get_buffer_samples

ALPHA = 1e-3
BETA = 1
GAMMA = 1
SAMPLES_PER_TASK = 100
IBTA_BATCH_SIZE = 32
IBDA_BATCH_SIZE = 64
UNSAMPLED_BUFFER_SIZE = 200

def train_main(model, train_loader, buffer, unsampled, epoch, epochs, device='cuda', alpha=ALPHA, beta=BETA, gamma=GAMMA, buffer_batch_size=IBTA_BATCH_SIZE, unsampled_batch_size=IBDA_BATCH_SIZE):
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    epoch_loss = 0

    for x, y_true in train_loader:
        optimizer.zero_grad()
        total_loss = 0
        x, y_true = x.to(device), y_true.to(device)
        z = model.encoder(x, stochastic=False)
        y_pred = model.classifier(z)
        n_loss = criterion(y_pred, y_true)

        ibta_loss = 0
        if len(buffer) >= buffer_batch_size:
            buffer_sample = make_dataloader_from_buffer(buffer=buffer, batch_size=buffer_batch_size, shuffle=True)
            x_buf, y_buf = next(iter(buffer_sample))
            x_buf, y_buf = x_buf.to(device), y_buf.to(device)
            z, mu, logvar = model.encoder(x_buf, stochastic=True)
            y_pred = model.classifier(z)
            ce_loss = criterion(y_pred, y_buf)
            kl_loss = alpha * (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / buffer_batch_size)
            ibta_loss = gamma * ce_loss + kl_loss

        ibds_loss = 0
        if len(unsampled) >= unsampled_batch_size:
            unsampled_sample = make_dataloader_from_buffer(buffer=unsampled, batch_size=unsampled_batch_size, shuffle=True)
            z_u_batch, y_u_batch = next(iter(unsampled_sample))
            z_u_batch, y_u_batch = z_u_batch.to(device), y_u_batch.to(device)
            y_pred = model.classifier(z_u_batch).squeeze(1)
            eta = 0.5 * (1 + math.cos(math.pi * epoch / epochs))
            ibds_loss = eta * beta * F.cross_entropy(y_pred, y_u_batch)

        total_loss = n_loss + ibta_loss + ibds_loss
        total_loss.backward()
        optimizer.step()
        epoch_loss += total_loss

    print(f'loss: {epoch_loss / len(train_loader)}')
    print('---')

def train_decoupling(sampled, unsampled, model, cos_loss_weight=1.0, device='cuda'):
    model.train()
    optimizer = torch.optim.SGD(model.encoder.parameters(), lr=0.01)
    total_loss = 0.0
    count = 0

    x_s_batch, y_s_batch = zip(*sampled)
    x_s_batch = torch.stack(x_s_batch).to(device)
    z_s_all = model.encoder(x_s_batch, stochastic=False)
    y_s_batch = torch.tensor(y_s_batch).to(device)

    for x_u, y_u in unsampled:
        x_u = x_u.to(device)
        z_u = model.encoder(x_u.unsqueeze(0), stochastic=False)
        class_sim = (y_s_batch == y_u)
        if class_sim.sum() == 0:
            continue
        z_class = z_s_all[class_sim]
        cos_sim = F.cosine_similarity(z_u, z_class)
        total_loss += cos_loss_weight * cos_sim.mean()
        count += 1

    decoupling_loss = total_loss / count
    optimizer.zero_grad()
    decoupling_loss.backward()
    optimizer.step()

def compute_accuracy(output, target):
    predicted_digits = output.argmax(1)
    correct_ones = (predicted_digits == target).type(torch.float)
    return correct_ones.sum().item()

def test(test_loader, model, device):
    model.eval()
    num_items = len(test_loader.dataset)
    test_loss = 0
    total_correct = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            z = model.encoder(data, stochastic=False)
            output = model.classifier(z)
            loss = criterion(output, target)
            test_loss += loss.item()
            total_correct += compute_accuracy(output, target)
    accuracy = total_correct / num_items
    return accuracy

def train(model, epochs, train_loader, buffer, unsampled_z, device='cuda'):
    model.to(device)
    for epoch in range(epochs):
        train_main(model=model, train_loader=train_loader, buffer=buffer, unsampled=unsampled_z, epoch=epoch, epochs=epochs, device=device)

    sampled, unsampled = get_buffer_samples(train_loader=train_loader, sample_count=SAMPLES_PER_TASK)
    buffer.extend(sampled)
    print(f'buffer size: {len(buffer)}')

    unsampled_batch = unsampled[:UNSAMPLED_BUFFER_SIZE]
    x_u_batch, y_u_batch = zip(*unsampled_batch)
    x_u_batch = torch.stack(x_u_batch).to(device)
    with torch.no_grad():
        z_batch = model.encoder(x_u_batch, stochastic=False).detach()
    z_u = list(zip(z_batch, y_u_batch))
    unsampled_z.extend(z_u)
    print(f'unsampled features size: {len(unsampled_z)}')

    train_decoupling(sampled=sampled, unsampled=unsampled[:UNSAMPLED_BUFFER_SIZE], model=model)

def train_continual(model, tasks, task_count, epochs, device='cuda'):
    running_test_accs = {i: [] for i in range(task_count)}
    buffer = []
    unsampled_z = []

    for task_id, task in tasks.items():
        print(f'training task: {task_id}')
        train_loader = task['train']
        train(model=model, epochs=epochs, train_loader=train_loader, buffer=buffer, unsampled_z=unsampled_z)
        for test_task_id in range(task_count):
            test_acc = 0 if test_task_id > task_id else test(test_loader=tasks[test_task_id]['test'], model=model, device=device)
            running_test_accs[test_task_id].append(test_acc)
    return running_test_accs