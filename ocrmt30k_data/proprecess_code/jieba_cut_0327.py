import jieba
import re
#train data

data_path="../pro_res/res/"
data_list=['train','valid','test']
lang_list=['zh_pseudo_','zh_']
for data in data_list:
    for lang in lang_list:
        fR = open(data_path+"ocr_"+lang+data+'.txt', 'r', encoding='utf-8')
        fW = open(data_path+data+'.tok.'+lang, 'w', encoding='UTF-8')
        index=0
        dulpnum=0
        sents=fR.readlines()
        print(len(sents))
        for sent in sents:
            index+=1
            if index % 10000==0:
                print(index)

            # sent = fR.readline()

            sent = sent.strip()
            # if not sent:
            #     # dulpnum+=1
            #     # # if data=='test':
            #     # #     print(index)
            #     # if dulpnum>=10:
            #     #     break
            #     # print("have null")
            #     # continue
            # dulpnum=0
            sent_list = jieba.cut(sent)
            sent_list=' '.join(sent_list)
            fW.write("%s\n" % sent_list)

        fR.close()
        fW.close()
