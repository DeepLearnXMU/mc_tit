import argparse
import functools
import os
import re

data_dir='../v2.0/'
# past_pseudo_dir='/paddle/ocr_data/0102_pseudo/'

# poly_flag=False #是否为多边形
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--mix_text_path", type=str, default="ocr_pseudo_cor_mix/")

args = parser.parse_args()
mix_text_path=data_dir+args.mix_text_path

# mix_text_path=past_pseudo_dir+'ocr_pseudo_0.5/'
trans_par_path = data_dir+'all_labeled_v2/'
match_text_path = data_dir+'ocr_pseudo_match/'

if not os.path.exists(match_text_path):
    os.makedirs(match_text_path)

mix_text_p = os.listdir(mix_text_path)
mix_text_p.sort()

transp = os.listdir(trans_par_path)
transp.sort()

global orig_num
orig_num=0
global final_num
final_num=0

class Ocr_origsent(object):
    name ='Ocr_origsent'
 
    def __init__(self,mix_index,ocr_text,sub_str_position=0,index=0):#初始化类的属性
        self.mix_index=mix_index
        self.ocr_text=ocr_text
        self.sub_str_position=sub_str_position
        self.index=index

def match(mix_text_path,mix_text,trans_path):
    # print("mix_text_path",mix_text_path)
    global orig_num
    global final_num
    if not os.path.exists(mix_text_path+mix_text):
        return
    fR = open(mix_text_path+mix_text, 'r', encoding='UTF-8')
    fW= open(match_text_path+mix_text,'w', encoding='UTF-8')
    sentences = fR.readlines()
    ftrans = open(trans_par_path+trans_path, 'r', encoding="UTF-8")

    tr=ftrans.readlines()
    ftrans.seek(0)
    res_pseudo=['*']*len(tr)
    if len(tr)==0:
        return

    #tr,trans为翻译文本
    #mix_sent，sentences为pseudo|||orig文本
    concact_position=[-1]*len(sentences)
    last_substr_position_list=[]
    last_substr_position={}
    complete_matched_flag=[False]*len(tr)
    complete_matched_flag_mix_sent=[False]*len(sentences)
    for index,trans in enumerate(tr):
        trans_last_position=0
        last_substr_position_list.append(trans_last_position)
    
    #先把完全匹配的都匹配上
    for mix_index,sent in enumerate(sentences):
        trans=ftrans.readline()
        sent=sent.strip()
        if not sent or sent=='*':
            continue
        orig_num+=1
        mix_sent=sent.split('||')
    #两者相等则直接写入
    #目前训练集做了去重，那训练集只需要用到第一次匹配到的图片就行，
    #验证测试没有去重，用到所有匹配到的图片
        for index,trans in enumerate(tr):
            trans_list=trans.split('|||')
            if mix_sent[1]==trans_list[0] and complete_matched_flag[index]==False and complete_matched_flag_mix_sent[mix_index]==False:
                res_pseudo[index]=sent
                complete_matched_flag[index]=True
                complete_matched_flag_mix_sent[mix_index]=True
                # print("完全匹配")
                break
    #创建对象保存
    sentences_object_list=[]
    for mix_index,sent in enumerate(sentences):
        sent=sent.strip()
        # if not sent or sent=='*':
        #     continue
        c=Ocr_origsent(mix_index,sent)
        sentences_object_list.append(c)
    #再处理合并的
    for mix_index,sent in enumerate(sentences_object_list):
        mix_sent=sent.ocr_text.split('||')

        if not mix_sent[0] or mix_sent[0]=='*':
            continue
        # #orig文本修正
        # orig_sent=mix_sent[1]
        
        #做合并匹配,先匹配子串，存储各个子串在主串中的位置
        for index,trans in enumerate(tr):
            trans_list=trans.split('|||')
            a=mix_sent[1] in trans_list[0]
            b=complete_matched_flag[index]
            d=complete_matched_flag_mix_sent[mix_index]
            if mix_sent[1] in trans_list[0] and trans_list[0]!='*' and complete_matched_flag[index]==False and complete_matched_flag_mix_sent[mix_index]==False:
                #记录子串在主串中的位置
                #concact_position[mix_index]=trans_list[0].index(mix_sent[1])
                
                sent.sub_str_position=trans_list[0].index(mix_sent[1])
                sent.index=index

    #先根据位置排序
    sentences_object_list.sort(key=functools.cmp_to_key(lambda x, y: x.sub_str_position - y.sub_str_position))
    #再根据位置合并各个子串到主串里面
    for s in sentences_object_list:
        sent=s.ocr_text
        index=s.index
        mix_sent=sent.split('||')
        mix_index=s.mix_index
        if not sent or sent=='*':
            continue
        if complete_matched_flag[index]==False and complete_matched_flag_mix_sent[mix_index]==False:
            if res_pseudo[index]=='*':
                res_pseudo[index]=sent
            #如果已有则加上去
            elif mix_sent[1]!=tr[index].split('|||')[0]:
                res2=res_pseudo[index].split('||')
                res2[0]+=mix_sent[0]
                res2[1]+=mix_sent[1]
                res_pseudo[index]='||'.join(res2)
    #合并后和原来不一致的，删掉,因为某些行无OCR结果，但也合并了
    for index,trans in enumerate(tr):
            trans_list=trans.split('|||')
            res_111=res_pseudo[index].split('||')
            if res_111[0]!='*' and res_111[1].replace(" ", "")!=trans_list[0].replace(" ", ""):
                res_pseudo[index]="*"
    final_num+=len(res_pseudo)
    for res in res_pseudo:
        fW.write("%s\n"%res)
    fR.close()
    fW.close()

print('len mix_text_p:',len(mix_text_p))
print('len transp:',len(transp))
for trans_path in transp:
    if not os.path.exists(trans_par_path+trans_path):
        continue
    mix_text=trans_path
    match(mix_text_path,mix_text,trans_path)
print(orig_num)
print(final_num)


