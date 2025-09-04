# 🧹 NOTEBOOK-SAFE Repository Cleanup Plan

## Critical Training Outputs to Preserve

### From GoEmotions_DeBERTa_Efficient_Workflow.ipynb:
- `./phase1_bce/` → Keep in root (notebook references)
- `./phase1_asymmetric/` → Keep in root (notebook references) 
- `./phase1_combined_07/` → Keep in root (notebook references)
- `./phase1_combined_05/` → Keep in root (notebook references)
- `./phase1_combined_03/` → Keep in root (notebook references)
- `./phase2_*/` → Keep in root (notebook references)
- `./final_*/` → Keep in root (notebook references)

### From GoEmotions_DeBERTa_Local.ipynb:
- `./test_single_run/` → Keep in root (notebook references)
- `./outputs/bce_baseline/` → Keep in root (notebook references)
- `./outputs/asymmetric_loss/` → Keep in root (notebook references)
- `./outputs/combined_loss_07/` → Keep in root (notebook references)

### Existing Structure (Keep):
- `models/` → Already correct
- `data/` → Already correct
- `logs/` → Already correct
- `notebooks/` → Already correct
- `scripts/` → Already correct
- `docs/` → Already correct

## Directories to MOVE to logs/:
- `rigorous_experiments/` → `logs/rigorous_experiments/`
- `samo_deberta_recovery/` → `logs/samo_deberta_recovery/`
- `deberta_out/` → `logs/deberta_out/`
- `samo_out/` → `logs/samo_out/`

## Directories to MOVE to models/:
- None (already organized)

## Test/Debug Directories to MOVE to logs/:
- `test_debug/` → `logs/test_debug/`
- `test_debug_single/` → `logs/test_debug_single/`
- `test_memory_fix/` → `logs/test_memory_fix/`

## Files to MOVE to docs/:
- `execution_issue_diagnosis_and_fix.md` → `docs/`
- `execution_readiness_assessment.md` → `docs/`
- `local_cache_verification_report.md` → `docs/`
- `performance_validation_framework.md` → `docs/`
- `reproducibility_documentation_framework.md` → `docs/`
- `results_analysis_framework.md` → `docs/`
- `strategic_execution_plan.md` → `docs/`

## Files to DELETE:
- `=2.6.0` (stray file)
- `debug_predictions.py` (if not needed)
- `phase1_summary.json` → Move to `logs/`

## Directories to DELETE (after verification):
- `Sync/` (if empty or not needed)
- Any empty phase1_* directories without content

## SAFETY MEASURES:
1. ✅ NO changes to notebook-referenced paths
2. ✅ Preserve all training outputs in original locations
3. ✅ Only move logs and documentation
4. ✅ Test notebooks after cleanup
