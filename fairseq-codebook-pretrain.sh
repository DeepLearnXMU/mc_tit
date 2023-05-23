export CUDA_VISIBLE_DEVICES=0
fairseq-train \
    data-bin/wmt22_zh_en_bpe_codebook \
    --arch dvae_multimodal_transformer_base \
    --task multimodal_translation \
    --codebook-size 2048 \
    --criterion multimodal_label_smoothed_cross_entropy_dvae_pretrain \
    --transalation-checkpoint checkpoints/wmt22_zh_en/checkpoint_best.pt \
    -s zh -t en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 8000 \
    --dropout 0.1 --weight-decay 0.0001 \
    --label-smoothing 0.1 \
    --max-tokens 8192 \
    --update-freq 4 \
    --save-dir checkpoints_new/wmt22_zh_en_codebook_pretrain \
    --save-interval-updates 2000 \
    --keep-best-checkpoints 5 \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --best-checkpoint-metric cor_construct_loss \
    --log-format simple \
    --patience 10 --max-source-positions 256 --max-target-positions 256 \
    --find-unused-parameters \
    --clip-norm 3
