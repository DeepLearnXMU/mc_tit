export CUDA_VISIBLE_DEVICES=0
#做ocr和匹配
python ocr_flow.py >./log/all_0.5.log

#将没匹配上的正确文本拼接上去
python mix_cor_text.py

#合并匹配标注文本
python match.py

root_dir=../v2.0
rm -rf $root_dir/pre_res/*
mkdir -p $root_dir/pre_res/res/tmp

#把数据都合到一个文件里面
python cat_file.py

cp ../bpe/zhen.bpe $root_dir/pre_res/res/tmp

#划分训练集等
python split_train_valid_test.py

#做分词+bpe
bash tok_bpe.sh
