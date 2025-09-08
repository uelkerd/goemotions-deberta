#!/bin/bash

# 🚀 GoEmotions DeBERTa Google Drive Backup Script
# Organized backup of training artifacts to TechLabs-2025/Final_Project/TRAINING/

set -e  # Exit on any error

# Configuration
REMOTE="drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/GoEmotions-DeBERTa-Backup"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

echo "🎯 GoEmotions DeBERTa - Google Drive Backup"
echo "=========================================="
echo "Target: $REMOTE"
echo "Timestamp: $TIMESTAMP"
echo

# Function to show progress
show_progress() {
    echo "✅ $1 completed"
}

# Function to backup with rate limiting (avoid quota issues)
backup_with_rate_limit() {
    local source="$1"
    local dest="$2"
    local desc="$3"

    echo "📤 Backing up $desc..."
    if [ -d "$source" ] || [ -f "$source" ]; then
        rclone --drive-pacer-min-sleep=1s --drive-pacer-burst=10 copy "$source" "$dest"
        show_progress "$desc backup"
    else
        echo "⚠️  $desc not found, skipping..."
    fi
}

# Create backup subdirectories
echo "📁 Creating backup directory structure..."
rclone mkdir "$REMOTE/models_$TIMESTAMP"
rclone mkdir "$REMOTE/outputs_$TIMESTAMP"
rclone mkdir "$REMOTE/logs_$TIMESTAMP"
rclone mkdir "$REMOTE/metadata_$TIMESTAMP"
show_progress "Directory structure"

# Backup models (most important)
backup_with_rate_limit "models/" "$REMOTE/models_$TIMESTAMP/" "models directory"

# Backup training outputs (checkpoints)
backup_with_rate_limit "outputs/" "$REMOTE/outputs_$TIMESTAMP/" "training outputs"

# Backup logs
backup_with_rate_limit "logs/" "$REMOTE/logs_$TIMESTAMP/" "training logs"

# Backup metadata and configs
backup_with_rate_limit "data/" "$REMOTE/metadata_$TIMESTAMP/data/" "dataset files"
backup_with_rate_limit "*.json" "$REMOTE/metadata_$TIMESTAMP/" "configuration files"
backup_with_rate_limit "*.py" "$REMOTE/metadata_$TIMESTAMP/scripts/" "Python scripts"

# Create backup manifest
echo "📋 Creating backup manifest..."
MANIFEST_FILE="/tmp/backup_manifest_$TIMESTAMP.txt"
cat > "$MANIFEST_FILE" << EOF
GoEmotions DeBERTa Backup Manifest
==================================
Backup Date: $(date)
Backup Timestamp: $TIMESTAMP

BACKUP CONTENTS:
---------------
Models: models/ → models_$TIMESTAMP/
Outputs: outputs/ → outputs_$TIMESTAMP/
Logs: logs/ → logs_$TIMESTAMP/
Metadata: configs, scripts, data → metadata_$TIMESTAMP/

IMPORTANT NOTES:
---------------
- Code/notebooks are backed up in GitHub
- This backup contains only training artifacts
- Models are the most critical for recovery
- Outputs contain training checkpoints
- Logs contain training metrics and history

RECOVERY INSTRUCTIONS:
---------------------
1. Download models from Google Drive
2. Clone repository from GitHub
3. Copy models to local models/ directory
4. Resume training or inference as needed
EOF

# Upload manifest
rclone copy "$MANIFEST_FILE" "$REMOTE/"

# Cleanup
rm "$MANIFEST_FILE"

echo
echo "🎉 BACKUP COMPLETE!"
echo "=================="
echo "📍 Location: $REMOTE"
echo "🗂️  Contents:"
echo "   ├── models_$TIMESTAMP/     (trained models)"
echo "   ├── outputs_$TIMESTAMP/    (training checkpoints)"
echo "   ├── logs_$TIMESTAMP/       (training logs)"
echo "   ├── metadata_$TIMESTAMP/   (configs, scripts, data)"
echo "   └── backup_manifest.txt    (this manifest)"
echo
echo "💡 Next Steps:"
echo "1. Verify backup: rclone ls \"$REMOTE\""
echo "2. Download if needed: rclone copy \"$REMOTE/models_$TIMESTAMP/\" ./models/"
echo "3. Resume training from checkpoints if desired"
echo
echo "✅ All training artifacts safely backed up to Google Drive!"
