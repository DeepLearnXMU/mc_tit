export CUDA_VISIBLE_DEVICES=0
img_feat_path=

fairseq-train \
    data-bin/ocr30k_data_merge \
    --arch multimodal_transformer_base \
    --task multimodal_translation \
    --criterion multimodal_label_smoothed_cross_entropy \
    --image-feat-path $img_feat_path \
    --img-feat-dim 768 \
    --codebook-size 2048 \
    --text-commit-weight 0.25\
    --img-commit-weight 0.75\
    --finetune-from-model checkpoints/wmt22_zh_en_codebook_ita_pretrain/checkpoint_last.pt \
    -s zh -t en \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --label-smoothing 0.1 \
    --max-tokens 2048 \
    --update-freq 2 \
    --save-dir checkpoints/final_tit\
    --keep-best-checkpoints 5 \
    --keep-interval-updates 5 \
    --no-epoch-checkpoints \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --best-checkpoint-metric bleu \
    --maximize-best-checkpoint-metric --log-format simple \
    --max-update 20000 --max-target-positions 256 \
    --seed 74 \
    --find-unused-parameters