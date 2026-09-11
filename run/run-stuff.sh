python3 -m scripts.eval_stage1_recall \
    --log_dir "/mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall_nl_table_description_20260908_132428" \
    --input_dir "/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input" \
    --output_dir "/mnt/labstore/fwolf/.logs/eval_stage1_recall_logs"

python3 -m scripts.eval_stage1_recall \
    --log_dir "/mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall_nl_table_description_without_filter_20260908_183458" \
    --input_dir "/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input" \
    --output_dir "/mnt/labstore/fwolf/.logs/eval_stage1_recall_logs"

python3 -m scripts.eval_stage1_recall \
    --log_dir "/mnt/labstore/fwolf/.logs/wikidata-federated-overlap-dataset/eval_stage1_recall_nl_table_description_without_filter_knee_20260909_123736" \
    --input_dir "/mnt/labstore/nzeitz/dataset/wikidata-federated-overlap-dataset/input" \
    --output_dir "/mnt/labstore/fwolf/.logs/eval_stage1_recall_logs"
