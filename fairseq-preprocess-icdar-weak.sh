fairseq-preprocess --source-lang zh --target-lang zh \
    --trainpref data/icdar19_lsvt_weak/train.tok.lower --validpref data/icdar19_lsvt_weak/valid.tok.lower \
    --destdir data-bin/icdar19_lsvt_weak \
    --thresholdsrc 0 --thresholdtgt 0 \
    --srcdict data-bin/wmt22_zh_en_bpe/dict.zh.txt \
    --tgtdict data-bin/wmt22_zh_en_bpe/dict.zh.txt \
    --workers 20

cp ./data-bin/icdar19_lsvt_weak/train.zh-zh.zh.bin ./data-bin/icdar19_lsvt_weak/train.zh-zh.zh-c.bin
cp ./data-bin/icdar19_lsvt_weak/train.zh-zh.zh.idx ./data-bin/icdar19_lsvt_weak/train.zh-zh.zh-c.idx

cp ./data-bin/icdar19_lsvt_weak/valid.zh-zh.zh.bin ./data-bin/icdar19_lsvt_weak/valid.zh-zh.zh-c.bin
cp ./data-bin/icdar19_lsvt_weak/valid.zh-zh.zh.idx ./data-bin/icdar19_lsvt_weak/valid.zh-zh.zh-c.idx