@echo off
for /L %%i in (0,1,4) do (
    python bii.py ^
        --mode infer ^
        --checkpoint_path saved_models/bii_new_dmpnn_balanced_run_%%i.pt ^
        --inference_input inference_outputs/gbm_prediction.csv ^
        --inference_output inference_outputs/BII/dmpnn_bii_new_dmpnn_balanced_run_%%i.csv
)
