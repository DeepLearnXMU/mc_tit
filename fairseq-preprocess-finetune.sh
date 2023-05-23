fairseq-preprocess --source-lang zh --target-lang en \
    --trainpref data/ocr30k_data_merge/train.tok.lower --validpref data/ocr30k_data_merge/valid.tok.lower \
    --testpref data/ocr30k_data_merge/test.tok.lower \
    --destdir data-bin/ocr30k_bpe_merge_c \
    --thresholdsrc 0 --thresholdtgt 0 \
    --srcdict data-bin/wmt22_zh_en_bpe/dict.zh.txt \
    --tgtdict data-bin/wmt22_zh_en_bpe/dict.en.txt \
    --workers 20

fairseq-preprocess --source-lang zh-c --target-lang en \
    --trainpref data/ocr30k_data_merge/train.tok.lower --validpref data/ocr30k_data_merge/valid.tok.lower \
    --testpref data/ocr30k_data_merge/test.tok.lower \
    --destdir data-bin/ocr30k_bpe_merge \
    --thresholdsrc 0 --thresholdtgt 0 \
    --srcdict data-bin/wmt22_zh_en_bpe/dict.zh.txt \
    --tgtdict data-bin/wmt22_zh_en_bpe/dict.en.txt \
    --workers 20

cp data-bin/ocr30k_bpe_merge_c/train.zh-c-en.zh-c.bin data-bin/ocr30k_data_merge/train.zh-en.zh-c.bin
cp data-bin/ocr30k_bpe_merge_c/train.zh-c-en.zh-c.idx data-bin/ocr30k_data_merge/train.zh-en.zh-c.idx