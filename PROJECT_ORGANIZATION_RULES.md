# Project Organization Rules

## Directory Structure Standards

### Root Directory
**ONLY** the following items are allowed in the root directory:
- Subdirectories (folders)
- README.md
- .gitignore
- .cursorrules
- PROJECT_ORGANIZATION_RULES.md (this file)

### Required Directory Structure

```
goemotions-deberta/
├── README.md
├── .gitignore
├── .cursorrules
├── PROJECT_ORGANIZATION_RULES.md
├── src/                          # All Python source code
│   ├── __init__.py
│   ├── models/                   # Model definitions and training code
│   ├── data/                     # Data processing and loading utilities
│   ├── evaluation/               # Evaluation and testing scripts
│   ├── inference/                # Inference and deployment scripts
│   └── utils/                    # Utility functions and helpers
├── scripts/                      # All shell scripts and automation
│   ├── training/                 # Training scripts
│   ├── evaluation/               # Evaluation scripts
│   ├── deployment/               # Deployment scripts
│   └── maintenance/              # Cleanup and maintenance scripts
├── configs/                      # Configuration files
│   ├── training/                 # Training configurations
│   ├── model/                    # Model configurations
│   └── deployment/               # Deployment configurations
├── data/                         # Raw and processed datasets
│   ├── raw/                      # Original datasets
│   ├── processed/                # Preprocessed datasets
│   └── external/                 # External data sources
├── outputs/                      # All generated outputs
│   ├── checkpoints/              # Model checkpoints
│   ├── logs/                     # Training and execution logs
│   ├── results/                  # Evaluation results
│   ├── plots/                    # Generated plots and visualizations
│   └── models/                   # Saved models
├── notebooks/                    # Jupyter notebooks
│   ├── exploration/              # Data exploration notebooks
│   ├── experiments/              # Experimental notebooks
│   └── tutorials/                # Tutorial and demo notebooks
├── docs/                         # Documentation
│   ├── api/                      # API documentation
│   ├── guides/                   # User guides
│   └── architecture/             # System architecture docs
├── tests/                        # Test files
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   └── fixtures/                 # Test data and fixtures
├── deployment/                   # Deployment configurations
├── temp/                         # Temporary files (gitignored)
└── logs/                         # Application logs (gitignored)
```

## File Organization Rules

### Python Files
- **ALL** Python files (.py) must be in `src/` or its subdirectories
- Use descriptive module names that indicate purpose
- Group related functionality into subdirectories
- Include `__init__.py` files in all Python packages

### Shell Scripts
- **ALL** shell scripts (.sh) must be in `scripts/` or its subdirectories
- Use descriptive names that indicate purpose
- Group by functionality (training, evaluation, deployment, maintenance)

### Configuration Files
- **ALL** configuration files (.json, .yaml, .toml, .txt) must be in `configs/` or its subdirectories
- Use descriptive names that indicate purpose
- Group by functionality (training, model, deployment)

### Output Files
- **ALL** generated files must be in `outputs/` or its subdirectories
- Never commit large output files to git
- Use descriptive subdirectories for different types of outputs

### Notebooks
- **ALL** Jupyter notebooks (.ipynb) must be in `notebooks/` or its subdirectories
- Use descriptive names that indicate purpose
- Group by functionality (exploration, experiments, tutorials)

## Naming Conventions

### Files
- Use snake_case for all file names
- Use descriptive names that indicate purpose
- Include version numbers when appropriate (e.g., `model_v2.py`)

### Directories
- Use snake_case for all directory names
- Use descriptive names that indicate purpose
- Keep names short but clear

### Scripts
- Prefix with action: `train_`, `evaluate_`, `deploy_`, `clean_`
- Include model or dataset name when relevant
- Use descriptive suffixes for variants

## Path Management

### Absolute vs Relative Paths
- Use relative paths within the project structure
- Always use `os.path.join()` or `pathlib.Path` for path construction
- Define project root as a constant at the top of scripts

### Import Paths
- Use absolute imports from project root
- Add project root to Python path in scripts
- Use consistent import structure across all modules

## Git and Version Control

### .gitignore Rules
- Ignore all files in `temp/` and `logs/` directories
- Ignore large output files (checkpoints, models, datasets)
- Ignore Python cache files (`__pycache__/`, `*.pyc`)
- Ignore IDE and editor files

### Commit Practices
- Commit organization changes separately from code changes
- Use descriptive commit messages
- Keep commits focused and atomic

## Maintenance Rules

### Regular Cleanup
- Run cleanup scripts weekly to remove temporary files
- Archive old outputs to external storage
- Remove unused dependencies and files

### Documentation
- Update this file when adding new directory structures
- Document any deviations from these rules
- Keep README.md updated with current structure

## Enforcement

### Pre-commit Hooks
- Set up pre-commit hooks to enforce file placement
- Validate directory structure before commits
- Check for files in wrong locations

### Code Review
- Always check file placement in code reviews
- Reject PRs that violate organization rules
- Suggest proper file placement for misplaced files

## Migration Guidelines

### When Adding New Files
1. Determine the appropriate directory based on file type and purpose
2. Create subdirectories if needed
3. Move file to correct location
4. Update any import statements or path references
5. Test that all paths work correctly

### When Reorganizing
1. Plan the new structure before making changes
2. Update all path references in scripts
3. Test thoroughly after reorganization
4. Update documentation
5. Commit changes atomically

## Examples

### Good File Placement
```
src/models/deberta_emotion_classifier.py  # Model definition
scripts/training/train_deberta.sh         # Training script
configs/training/deberta_config.json      # Training configuration
outputs/checkpoints/deberta_epoch_5.pt    # Model checkpoint
notebooks/experiments/emotion_analysis.ipynb  # Analysis notebook
```

### Bad File Placement
```
deberta_model.py                          # Should be in src/models/
train.sh                                  # Should be in scripts/training/
config.json                               # Should be in configs/
checkpoint.pt                             # Should be in outputs/checkpoints/
analysis.ipynb                            # Should be in notebooks/
```

## Emergency Procedures

### If Files Are Misplaced
1. Identify the correct location based on these rules
2. Move the file to the correct location
3. Update all references to the file
4. Test that everything still works
5. Commit the fix

### If Structure Becomes Messy
1. Stop adding new files
2. Create a reorganization plan
3. Move files systematically
4. Update all references
5. Test thoroughly
6. Commit the reorganization

Remember: **A well-organized codebase is a maintainable codebase!**
