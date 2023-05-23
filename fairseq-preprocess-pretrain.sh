fairseq-preprocess --source-lang zh --target-lang en \
    --trainpref data/wmt22_zh_en_bpe/train.tok --validpref data/wmt22_zh_en_bpe/valid.tok --testpref data/wmt22_zh_en_bpe/test.tok\
    --destdir data-bin/wmt22_zh_en_bpe \
    --thresholdsrc 0 --thresholdtgt 0 \
    --joined-dictionary \
    --workers 20
