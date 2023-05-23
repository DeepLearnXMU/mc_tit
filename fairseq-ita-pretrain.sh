img_feat_path=

export CUDA_VISIBLE_DEVICES=0
fairseq-train \
    data-bin/icdar19_lsvt_weak \
    --arch dvae_multimodal_transformer_base \
    --task multimodal_translation \
    --criterion multimodal_label_smoothed_cross_entropy_match_pretrain \
    --image-feat-path $img_feat_path \
    --img-feat-dim 768 \
    --codebook-size 2048 \
    --finetune-from-model checkpoints/wmt22_zh_en_codebook_pretrain/checkpoint_best.pt \
    -s zh -t zh \
    --share-all-embeddings \
    --optimizer adam --adam-betas '(0.9, 0.98)' \
    --lr 5e-5 --lr-scheduler inverse_sqrt --warmup-updates 2000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --label-smoothing 0.1 \
    --max-tokens 4096 \
    --update-freq 1 \
    --save-dir checkpoints_ocr30k/wmt22_zh_en_codebook_ita_pretrain\
    --save-interval-updates 2000 \
    --no-epoch-checkpoints \
    --log-format simple \
    --max-update 20000 --max-target-positions 256 \
    --find-unused-parameters 
