import argparse
from email.policy import default
import os
import re

def main():
    # parser = argparse.ArgumentParser()
    # # parser.add_argument("img_par_path", type=str,default='/home/ocr_data_raw/flow2/image-filter-2/')
    # # parser.add_argument("trans_par_path", type=str,default='/home/ocr_data_raw/flow2/trans2_utf8/')
    # # parser.add_argument("mix_text_path", type=str,default='/home/ocr_data_raw/flow2/OCR_pseudo_2/')
    # # parser.add_argument("fImg", type=str,default='/home/ocr_data_raw/ocr_data_v2/image.txt')
    # # parser.add_argument("fzh_pseudo", type=str,default='/home/ocr_data_raw/ocr_data_v2/ocr_zh_pseudo.txt')
    # # parser.add_argument("fzh_orig", type=str,default='/home/ocr_data_raw/ocr_data_v2/ocr_zh.txt')
    # # parser.add_argument("fen", type=str,default='/home/ocr_data_raw/ocr_data_v2/ocr_en.txt')

    data_dir='../v2.0/'
    img_par_path =data_dir+'whole_image_v2/'
    trans_par_path = data_dir+'all_labeled_v2/'
    match_text_path = data_dir+'ocr_pseudo_match/'
    res_dir='../pro_res/'
    
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    fImg = open(res_dir+'image.txt', 'w', encoding='UTF-8')
    fzh_pseudo= open(res_dir+'ocr_zh_pseudo.txt', 'w', encoding='UTF-8')
    fzh_orig = open(res_dir+'ocr_zh.txt', 'w', encoding='UTF-8')
    fen = open(res_dir+'ocr_en.txt', 'w', encoding='UTF-8')
    
    items = os.listdir(img_par_path)
    items.sort()

    transp = os.listdir(trans_par_path)
    transp.sort()

    mix_text_p = os.listdir(match_text_path)
    mix_text_p.sort()
    
    print('len items:',len(items))
    print('len transp:',len(transp))
    print('len mix_text_p:',len(mix_text_p))

    for trans_path in transp:
        img_path=trans_path[:-4]+'.jpg'
        mix_text=trans_path
        if not os.path.exists(match_text_path+mix_text):
            continue
        fR = open(match_text_path+mix_text, 'r', encoding='UTF-8')
        sentences = fR.readlines()
        ftrans = open(trans_par_path+trans_path, 'r', encoding="UTF-8")
        tr=ftrans.readlines()
        ftrans.seek(0)
        # if(len(sentences)!=len(t)):
        #     continue
        #print(trans_path)
        for sent in sentences:
            trans=ftrans.readline()
            sent=sent.strip()
            #print(sent)
            if not sent or sent=='*':
                continue
            mix_sent=sent.split('||')
            pattern="[\u4e00-\u9fa5]+"#中文正则表达式
            reg = re.compile(pattern) #生成正则对象
            #找有没有中文
            results=reg.findall(mix_sent[1])
            if not results or not trans:
                continue
            trans_list=trans.split('|||')
            #对于修正了ocr错误的文本，标注格式为：
            # “原始标注|||修正后标注|||译文”
            # 只保留“修正后标注|||译文”
            if len(trans_list)>=3:
                trans_list=trans_list[1:]
            if len(trans_list)<2:
                trans_list=trans.split('||')
            if len(trans_list)==1:
                print(trans_path)
                continue
            # if len(mix_sent[0])/len(mix_sent[1])<0.3 or len(mix_sent[0])/len(mix_sent[1])>3:
            #     continue
            fzh_pseudo.write("%s\n" % mix_sent[0])
            fzh_orig.write("%s\n" % trans_list[0])
            fen.write("%s\n" % trans_list[1].strip())
            fImg.write("%s\n" % img_path)
        fR.close()
        ftrans.close()
    fImg.close()
    fen.close()
if __name__=="__main__":
    main()