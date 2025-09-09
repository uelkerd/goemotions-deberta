import json
import matplotlib.pyplot as plt
import numpy as np

configs = {
    'BCE_Parallel': './logs_backup/BCE_Parallel/checkpoint-2500/trainer_state.json',
    'Combined_05_Parallel': './logs_backup/Combined_05_Parallel/checkpoint-8794/trainer_state.json'
}

for config, file_path in configs.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    log_history = data['log_history']
    
    train_steps = []
    train_losses = []
    eval_steps = []
    eval_losses = []
    
    for entry in log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_steps.append(entry['step'])
            train_losses.append(entry['loss'])
        elif 'eval_loss' in entry:
            eval_steps.append(entry['step'])
            eval_losses.append(entry['eval_loss'])
    
    plt.figure(figsize=(10, 6))
    if train_steps:
        plt.plot(train_steps, train_losses, label='Train Loss', color='blue')
    if eval_steps:
        plt.plot(eval_steps, eval_losses, label='Eval Loss', color='red', marker='o')
    plt.title(f'Loss Curves for {config}')
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{config}_loss_curve.png')
    plt.close()
    print(f'Plot saved for {config}: {config}_loss_curve.png')

print('All plots saved.')
