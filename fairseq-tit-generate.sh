export CUDA_VISIBLE_DEVICES=0
img_feat_path=
fairseq-generate data-bin/ocr30k_data_merge \
    -s zh -t en \
    --arch multimodal_transformer_base \
    --task multimodal_translation \
    --image-feat-path $img_feat_path \
    --img-feat-dim 768 \
    --max-tokens 2048 \
    --path checkpoints/final_tit/checkpoint_best.pt \
    --beam 5 --remove-bpe \
    --max-len-a 1.2 --max-len-b 10 \
    --eval-bleu-detok moses \
    --sacrebleu 
