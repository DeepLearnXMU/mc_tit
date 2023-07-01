SRC=zh_
SRC_pseu=zh_pseudo_
TRG=en
#change dir to your own mosesdecoder and subword_nmt dir
mosesdecoder=/paddle/ocr_nmt/baseline_fairseq/fairseq/examples/translation/mosesdecoder

subword_nmt=/paddle/ocr_nmt/baseline_fairseq/fairseq/examples/translation/subword-nmt/subword_nmt

bpe_operations=10000

LC=$mosesdecoder/scripts/tokenizer/lowercase.perl

python jieba_cut_0327.py

data_path=../pro_res/res

#fine-tune ocr_data
#cut
for f in train valid test; do
    cat $data_path/ocr_$TRG"_"$f.txt | \
    $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -threads 8 -l $TRG | \
    $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -threads 8 -l $TRG > $data_path/$f.tok.$TRG
done
# clean empty and long sentences, and sentences with high source-target ratio (training corpus only)
# $mosesdecoder/scripts/training/clean-corpus-n.perl ./train.tok $SRC $TRG ./train.tok.clean 1 180

mkdir -p $data_path/before_bpe
mkdir -p $data_path/after_bpe
#全部转小写
for l in $SRC $SRC_pseu $TRG; do
    perl $LC < $data_path/train.tok.$l > $data_path/before_bpe/train.tok.lower.$l
    perl $LC < $data_path/valid.tok.$l > $data_path/before_bpe/valid.tok.lower.$l
    perl $LC < $data_path/test.tok.$l > $data_path/before_bpe/test.tok.lower.$l
done

mkdir -p $data_path/tmp
ocr_bpe=$data_path/tmp

# train BPE
cat $data_path/train.tok.$SRC $data_path/train.tok.$SRC_pseu $data_path/train.tok.$TRG | $subword_nmt/learn_bpe.py -s $bpe_operations > $ocr_bpe/$SRC$TRG.ocr.bpe

#wmt22 bpe
for L in $SRC $SRC_pseu $TRG; do
    for f in train.tok.lower.$L valid.tok.lower.$L test.tok.lower.$L; do
        echo "apply_bpe.py to ${f}..."
        $subword_nmt/apply_bpe.py -c $ocr_bpe/zhen.bpe < $data_path/before_bpe/$f > $data_path/after_bpe/$f
        # $subword_nmt/apply_bpe.py -c tmp/$SRC$TRG.bpe < ocr_data/train.zh > ocr_data/after_bpe/$f
    done
done

mkdir -p $data_path/after_ocr_bpe
#ocr bpe
for L in $SRC $SRC_pseu $TRG; do
    for f in train.tok.lower.$L valid.tok.lower.$L test.tok.lower.$L; do
        echo "apply_bpe.py to ${f}..."
        $subword_nmt/apply_bpe.py -c $ocr_bpe/zh_en.ocr.bpe < $data_path/before_bpe/$f > $data_path/after_ocr_bpe/$f
        # $subword_nmt/apply_bpe.py -c tmp/$SRC$TRG.bpe < ocr_data/train.zh > ocr_data/after_bpe/$f
    done
done

for f in train valid test;do
    mv $data_path/after_bpe/$f.tok.lower.zh_ $data_path/after_bpe/$f.tok.lower.zh-c
    mv $data_path/after_bpe/$f.tok.lower.zh_pseudo_ $data_path/after_bpe/$f.tok.lower.zh
done


for f in train valid test; do
cp $data_path/image_$f.txt $data_path/after_ocr_bpe
done

for f in train valid test; do
cp $data_path/image_$f.txt $data_path/after_bpe
done

# for f in valid test; do
# cp $save_dev_test/image_$f.txt $data_path/after_bpe
#     for l in en zh zh-c;do
#         cp $save_dev_test/$f.tok.lower.$l $data_path/after_bpe
#     done
# done
# TEXT=./ocr_data/after_bpe
# fairseq-preprocess --source-lang zh --target-lang en \
#     --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
#     --destdir data-bin/ocr_data.tokenized  --joined-dictionary


# #full wmt22 data

# cat ./wmt22_full/wmt22.$SRC | \
# $mosesdecoder/scripts/tokenizer/normalize-punctuation.perl -threads 8 -l $SRC | \
# $mosesdecoder/scripts/tokenizer/tokenizer.perl -a -threads 8 -l $SRC > ./wmt22_full/wmt22.tok.$SRC

# $mosesdecoder/scripts/training/clean-corpus-n.perl wmt22_full/wmt22.tok $SRC $TRG wmt22_full/wmt22.tok.clean 1 180

# perl $LC < wmt22_full/wmt22.tok.clean.$SRC > wmt22_full/wmt22.lower.$SRC

# # train BPE
# cat wmt22_full/wmt22.lower.$SRC wmt22_full/wmt22.tok.clean.$TRG | $subword_nmt/learn_bpe.py --num-workers 16 -s $bpe_operations > wmt22_full/$SRC$TRG.bpe

# # apply BPE

# $subword_nmt/apply_bpe.py -c wmt22_full/$SRC$TRG.bpe < wmt22_full/wmt22.lower.$SRC > wmt22_full/wmt22.bpe.$SRC
# $subword_nmt/apply_bpe.py -c wmt22_full/$SRC$TRG.bpe < wmt22_full/wmt22.tok.clean.$TRG > wmt22_full/wmt22.bpe.$TRG

# #dev data
# $subword_nmt/apply_bpe.py -c wmt22_full/$SRC$TRG.bpe < news2017.dev.lower.$SRC > wmt22_full/after_bpe/news2017.dev.bpe.$SRC
# $subword_nmt/apply_bpe.py -c wmt22_full/$SRC$TRG.bpe < news2017_jieba.dev.$TRG > wmt22_full/after_bpe/news2017.dev.bpe.$TRG

# #fine-tune data
# for L in $SRC $TRG; do
#     for f in train.tok.$L valid.tok.$L test.tok.$L; do
#         echo "apply_bpe.py to ${f}..."
#         $subword_nmt/apply_bpe.py -c wmt22_full/$SRC$TRG.bpe < ocr_data/before_bpe/$f > ocr_data/after_bpe_wmt22_full/$f
#         # $subword_nmt/apply_bpe.py -c tmp/$SRC$TRG.bpe < ocr_data/train.zh > ocr_data/after_bpe/$f
#     done
# done