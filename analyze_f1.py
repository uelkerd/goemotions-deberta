import json

configs = {
    'BCE_Parallel': './logs_backup/BCE_Parallel/checkpoint-2500/trainer_state.json',
    'Combined_05_Parallel': './logs_backup/Combined_05_Parallel/checkpoint-8794/trainer_state.json'
}

rare_emotions = ['awe', 'disgust', 'embarrassment', 'excitement', 'grief', 'pride', 'realization', 'relief']  # rare ones, awe as admiration

for config, file_path in configs.items():
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    log_history = data['log_history']
    # Find last eval entry
    last_eval = None
    for entry in reversed(log_history):
        if 'eval_loss' in entry:
            last_eval = entry
            break
    
    if last_eval:
        print(f"\nClass-wise F1 scores for {config} (final eval at step {last_eval.get('step', 'N/A')}):")
        print("Rare emotions F1:")
        for emotion in rare_emotions:
            key = f'eval_f1_{emotion}' if emotion != 'awe' else 'eval_f1_admiration'
            f1 = last_eval.get(key, 0.0)
            print(f"  {emotion}: {f1:.4f}")
        print(f"Macro F1: {last_eval.get('eval_f1_macro', 0.0):.4f}")
        print(f"Weighted F1: {last_eval.get('eval_f1_weighted', 0.0):.4f}")
    else:
        print(f"No eval entry found for {config}")
