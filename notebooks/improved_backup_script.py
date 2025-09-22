"""
Improved backup script for GoEmotions-DeBERTa model outputs
Correctly handles rclone paths and directory structure
"""

backup_script = """#!/bin/bash

# Backup script for GoEmotions-DeBERTa model outputs
# Syncs training outputs to Google Drive to prevent disk quota issues

# Target Google Drive folder - EXACT path from rclone config
DRIVE_TARGET="drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/GoEmotions-DeBERTa-Backup"

# Local paths to backup
MODEL_OUTPUTS="./outputs"
MODEL_CACHE="./models"
DATASET_CACHE="./data"
LOGS="./logs"

# Timestamp
TIMESTAMP=$(date +"%Y-%m-%d_%H-%M-%S")
echo "🔄 Starting backup at $TIMESTAMP"

# Check if rclone is configured
if ! rclone version &>/dev/null; then
    echo "❌ rclone not found or not configured"
    exit 1
fi

# Verify the remote exists and we can access it
if ! rclone lsd "drive:" &>/dev/null; then
    echo "❌ Cannot access 'drive:' remote. Check rclone configuration."
    exit 1
fi

# Check/create directory structure
echo "🔍 Verifying backup directory structure..."
if ! rclone lsd "$DRIVE_TARGET" &>/dev/null; then
    echo "🔨 Creating main backup directory..."
    rclone mkdir "$DRIVE_TARGET"
fi

# Create subdirectories
for dir in "outputs" "models" "data" "logs"; do
    if ! rclone lsd "$DRIVE_TARGET/$dir" &>/dev/null; then
        echo "🔨 Creating $DRIVE_TARGET/$dir"
        rclone mkdir "$DRIVE_TARGET/$dir"
    fi
done

# Backup eval reports first (highest value, smallest size)
echo "📊 Backing up evaluation reports..."
find "$MODEL_OUTPUTS" -name "eval_report.json" -type f | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    # Ensure parent directory exists
    rclone mkdir "$target_dir" 2>/dev/null
    
    echo "   $file → $target_path"
    rclone copy "$file" "$target_dir" --progress
done

# Backup model weights (most important for resuming)
echo "🤖 Backing up model weights..."
find "$MODEL_OUTPUTS" -name "pytorch_model.bin" -o -name "model.safetensors" | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    # Ensure parent directory exists
    rclone mkdir "$target_dir" 2>/dev/null
    
    echo "   $file → $target_path"
    rclone copy "$file" "$target_dir" --progress
done

# Backup model configs
echo "⚙️ Backing up model configs..."
find "$MODEL_OUTPUTS" -name "config.json" -o -name "special_tokens_map.json" -o -name "tokenizer_config.json" -type f | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    # Ensure parent directory exists
    rclone mkdir "$target_dir" 2>/dev/null
    
    echo "   $file → $target_path"
    rclone copy "$file" "$target_dir"
done

# Backup base model cache (smaller files first)
echo "💾 Backing up model metadata..."
find "$MODEL_CACHE" -name "*.json" -type f | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    rclone mkdir "$target_dir" 2>/dev/null
    rclone copy "$file" "$target_dir"
done

# Backup tokenizer files
echo "🔤 Backing up tokenizer files..."
find "$MODEL_CACHE" -name "*.model" -o -name "*.txt" -type f | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    rclone mkdir "$target_dir" 2>/dev/null
    rclone copy "$file" "$target_dir"
done

# Backup dataset files (important but can be large)
echo "📊 Backing up dataset metadata..."
find "$DATASET_CACHE" -name "*.json" -type f | while read -r file; do
    rel_path=$(echo "$file" | sed "s|^\./||")
    target_path="$DRIVE_TARGET/$rel_path"
    target_dir=$(dirname "$target_path")
    
    rclone mkdir "$target_dir" 2>/dev/null
    rclone copy "$file" "$target_dir"
done

# Backup logs
echo "📝 Backing up logs..."
rclone copy "$LOGS" "$DRIVE_TARGET/logs" --update

# Record backup in log file
echo "$TIMESTAMP: Backup completed successfully" >> "./backup_history.log"
rclone copy "./backup_history.log" "$DRIVE_TARGET/"

# Check disk space after backup
FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
USED_PERCENT=$(df -h . | awk 'NR==2 {print $5}')
echo "💾 Disk space after backup: $FREE_SPACE free ($USED_PERCENT used)"

echo "✅ Backup completed at $(date)"
"""

# Auto backup script that runs during training
auto_backup_script = """#!/bin/bash

# Auto backup script that runs during training
# Set to backup every 15 minutes to prevent disk quota issues

BACKUP_INTERVAL=900  # 15 minutes
BACKUP_SCRIPT="/home/user/goemotions-deberta/backup_to_drive.sh"

echo "🔄 Starting automatic backup service at $(date)"
echo "📁 Target: drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/GoEmotions-DeBERTa-Backup"
echo "⏱️ Backup interval: $BACKUP_INTERVAL seconds (15 minutes)"
echo "🧹 Auto-cleanup: Enabled for disk usage >85%"

while true; do
    # Run backup
    echo ""
    echo "🔄 Running scheduled backup ($(date))"
    bash "$BACKUP_SCRIPT"
    
    # Check for disk quota issues
    FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    USED_PERCENT=$(df -h . | awk 'NR==2 {print $5}')
    USED_NUM=${USED_PERCENT%\%}
    
    if [ "$USED_NUM" -gt 85 ]; then
        echo "⚠️ WARNING: High disk usage ($USED_PERCENT)"
        echo "🧹 Cleaning old checkpoints after backup..."
        # Find and list what will be removed first
        echo "Finding old checkpoints to remove..."
        find ./outputs -name "checkpoint-*" -type d | sort | head -n -5
        # Then remove them
        find ./outputs -name "checkpoint-*" -type d | sort | head -n -5 | xargs rm -rf
        
        # Check space after cleanup
        FREE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
        USED_PERCENT=$(df -h . | awk 'NR==2 {print $5}')
        echo "💾 Disk space after cleanup: $FREE_SPACE free ($USED_PERCENT used)"
    fi
    
    echo "💤 Next backup in $(($BACKUP_INTERVAL / 60)) minutes ($(date -d "+$BACKUP_INTERVAL seconds"))"
    sleep $BACKUP_INTERVAL
done
"""

# Function to test rclone connection
test_rclone_script = """#!/bin/bash

# Test rclone connection to Google Drive
echo "🔍 Testing rclone connection to Google Drive..."

# Check if rclone is installed
if ! command -v rclone &> /dev/null; then
    echo "❌ rclone not found. Please install it first."
    exit 1
fi

# Check if drive remote exists
if ! rclone listremotes | grep -q "drive:"; then
    echo "❌ 'drive:' remote not found in rclone configuration."
    echo "Please configure rclone with 'rclone config' command first."
    exit 1
fi

# Try to list root directory
echo "📁 Testing access to drive: remote..."
if ! rclone lsd drive: &>/dev/null; then
    echo "❌ Cannot access 'drive:' remote. Please check your rclone configuration."
    exit 1
fi

# Try to access or create the backup directory
TARGET_DIR="drive:00_Projects/🎯 TechLabs-2025/Final_Project/TRAINING/GoEmotions-DeBERTa-Backup"
echo "📁 Testing access to target directory: $TARGET_DIR"

if ! rclone lsd "$TARGET_DIR" &>/dev/null; then
    echo "🔨 Target directory doesn't exist. Creating it now..."
    if ! rclone mkdir "$TARGET_DIR" &>/dev/null; then
        echo "❌ Failed to create target directory. Please check permissions and path."
        exit 1
    else
        echo "✅ Target directory created successfully."
    fi
else
    echo "✅ Target directory exists and is accessible."
fi

# Try writing a test file
TEST_FILE="/tmp/rclone_test_$(date +%s).txt"
echo "This is a test file. Created at $(date)" > "$TEST_FILE"

echo "📤 Uploading test file to $TARGET_DIR..."
if ! rclone copy "$TEST_FILE" "$TARGET_DIR" &>/dev/null; then
    echo "❌ Failed to upload test file. Please check permissions."
    rm -f "$TEST_FILE"
    exit 1
else
    echo "✅ Test file uploaded successfully."
    
    # Try to read the file back
    echo "📥 Verifying test file..."
    if ! rclone ls "$TARGET_DIR/$(basename "$TEST_FILE")" &>/dev/null; then
        echo "⚠️ Warning: File uploaded but not immediately visible. This may be normal for Google Drive."
    else
        echo "✅ File verified in target location."
        # Clean up the test file
        rclone delete "$TARGET_DIR/$(basename "$TEST_FILE")" &>/dev/null
    fi
fi

rm -f "$TEST_FILE"
echo "✅ rclone connection test completed successfully!"
echo "✅ Backup system ready to use with target: $TARGET_DIR"
"""