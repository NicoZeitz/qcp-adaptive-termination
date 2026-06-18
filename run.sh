#!/usr/bin/env bash
uv run main.py \
    --name "Unknown" \
    --input "../data/unknown-synthetic/total_newest_population_rho_0.0_lambda_0.0.json" \
    --log_dir "../out-test" \
    --start_stage 1 \
    --end_stage 3 \
    --provider OpenAI