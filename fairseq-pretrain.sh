export CUDA_VISIBLE_DEVICES=0
fairseq-train \
    data-bin/wmt22_zh_en_bpe \
    --arch transformer \
    -s zh -t en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 20000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-tokens 16384 \
    --update-freq 2 \
    --save-dir checkpoints/wmt22_zh_en \
    --save-interval-updates 2000 \
    --keep-best-checkpoints 5 \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --log-format simple \
    --patience 10 --max-source-positions 256 --max-target-positions 256 \
    --fp16 \
    --find-unused-parameters