from collections import Counter
import os
import random
import shutil
import zhconv
def hant_2_sim(hant_str: str):
    '''
    Function: 将 hant_str 由繁体转化为简体
    '''
    return zhconv.convert(hant_str, 'zh-hans')

def copyfile(infile,outfile):
    try:
        shutil.copy(infile,outfile)
    except:
        print('''Can't open this file''')
        return
        
r=random.random
random.seed(7)

data_path="../pro_res/"
#dev_test_path='/paddle/label_data_2400-3600/dev_test.txt'
#dev_path='/paddle/ocr_data/flow_all_0103-0103-save/res/image_valid.txt'
#test_path='/paddle/ocr_data/flow_all_0103-0103-save/res/image_test.txt'
dev_path='../v2.0/after_proprecess/after_tok_bpe/image_valid.txt'
test_path='../v2.0/after_proprecess/after_tok_bpe/image_test.txt'
train_img_path='../v2.0/after_proprecess/after_tok_bpe/image_train.txt'

fImg = open(data_path+"image.txt", 'r', encoding='UTF-8')
#f_train_Img=open('/paddle/ocr_data/flow_all_0103-ocr_correct_mix/res/after_bpe/image_train.txt', 'r', encoding='UTF-8')
fzh_pseudo= open(data_path+"ocr_zh_pseudo.txt", 'r', encoding='UTF-8')
fzh_orig = open(data_path+"ocr_zh.txt", 'r', encoding='UTF-8')
fen = open(data_path+"ocr_en.txt", 'r', encoding='UTF-8')
#fdev_test=open(dev_test_path, 'r', encoding='UTF-8')
fdev=open(dev_path, 'r', encoding='UTF-8')
ftest=open(test_path, 'r', encoding='UTF-8')
f_train_img=open(train_img_path,'r',encoding='UTF-8')


image_list=fImg.readlines()
zh_pseudo_list=fzh_pseudo.readlines()
zh_orig_list=fzh_orig.readlines()
en_list=fen.readlines()
#dev_test_list=fdev_test.readlines()
dev_list=fdev.readlines()
test_list=ftest.readlines()
train_img_list=f_train_img.readlines()

image_set=set(image_list)
image_no_dulp_list=list(image_set)
image_no_dulp_list.sort()
# random.shuffle(image_no_dulp_list,random=r)
dataSize=len(image_set)

print("dataSize:",dataSize)
# for i in range(len(dev_test_list)):
#     dev_test_list[i]=dev_test_list[i].strip()
#     dev_test_list[i]+='.jpg\n'
# dev_test_set=set(dev_test_list)
# random.shuffle(dev_test_list,random=r)

# b = dict(Counter(dev_test_list))
# print ([key for key,value in b.items()if value > 1])  #只展示重复元素
# print ({key:value for key,value in b.items()if value > 1})  #展现重复元素和重复次数

# dev_list=[]

# for d in dev_test_set:
#     flag=False
#     for i in image_set:
#         if d==i:
#             dev_list.append(d)
#             flag=True
#     if flag==False:
#         print(d)
def find_diff_by_twolist(list1,list2):
    '''
    :param list1: 列表1
    :param list2: 列表2
    :return:
    '''
    from collections import Counter
    newli=list(set(list1)) + list(set(list2))
    count=Counter(newli)
    same, diff = [], []
    for i in count.keys():
        if(count.get(i)>=2):
            same.append(i)
        else:
            diff.append(i)
    # print("same is {},diff is {}".format(same, diff))
    return same,diff

# dev_list,train_list=find_diff_by_twolist(image_set,dev_test_set)
# # #找出里面不一样的元素
# # train_list=image_set.symmetric_difference(dev_test_set)
# train_list=list(image_set-set(dev_list))

# del_list=list(dev_test_set-set(dev_list))
# del_list.sort()

dev_list=list(dict.fromkeys(dev_list))
test_list=list(dict.fromkeys(test_list))
# train_list=f_train_Img.readlines()

train_list_orig=list(image_set-set(dev_list)-set(test_list))

for item in dev_list:
    if item not in image_set:
        print(item)
for item in test_list:
    if item not in image_set:
        print(item)
train_list=list(dict.fromkeys(train_img_list))

#diff_list = list(set(train_list_orig)-set(train_list))

# PAL07944
# gt_lsvt_003413 未被算在里面，下一个版本的，把这两个加上去

# new_list = list(dict.fromkeys(old_list))

# train_list=list(train_list)
# random.shuffle(dev_list,random=r)

# f_dev_list=open("/paddle/label_data_2400-3600/dev_test_1230.txt", 'w', encoding='UTF-8')
# f_dev_list.write("%s" % dev_list)

# validSize=int(len(dev_list)/2)
# testSize = len(dev_list)-validSize

# print(dev_list[1999])
validSize = 1000
testSize = 1000
trainSize= len(train_list)
#trainSize = dataSize - validSize - testSize
data_path+="res/"
idx=['train','valid','test']

part_image_flag=True
for n in idx:
    fImg_d = open(data_path+"image_"+n+".txt", 'w', encoding='UTF-8')
    fzh_pseudo_d= open(data_path+"ocr_zh_pseudo_"+n+".txt", 'w', encoding='UTF-8')
    fzh_orig_d = open(data_path+"ocr_zh_"+n+".txt", 'w', encoding='UTF-8')
    fen_d = open(data_path+"ocr_en_"+n+".txt", 'w', encoding='UTF-8')
    print(n)
    if(n=='train'):
        for i in range(trainSize):
            image_name = train_list[i]
            for id,name in enumerate(image_list):
                if name == image_name:
                    fImg_d.write("%s" % image_name)
                    fzh_orig_d.write("%s" % hant_2_sim(zh_orig_list[id]))
                    fen_d.write("%s" % en_list[id])
                    fzh_pseudo_d.write("%s" % hant_2_sim(zh_pseudo_list[id]))
    if(n=='valid'):
        for j in range(validSize):
            image_name = dev_list[j]
            for id,name in enumerate(image_list):
                if name == image_name:
                    fImg_d.write("%s" % image_name)
                    fzh_orig_d.write("%s" % hant_2_sim(zh_orig_list[id]))
                    fen_d.write("%s" % en_list[id])
                    fzh_pseudo_d.write("%s" % hant_2_sim(zh_pseudo_list[id]))
    
    if(n=='test'):
        print(j)
        for k in range(testSize):
            image_name = test_list[k]
            for id,name in enumerate(image_list):
                if name == image_name:
                    fImg_d.write("%s" % image_name)
                    fzh_orig_d.write("%s" % hant_2_sim(zh_orig_list[id]))
                    fen_d.write("%s" % en_list[id])
                    fzh_pseudo_d.write("%s" % hant_2_sim(zh_pseudo_list[id]))
    fImg_d.close()
    fzh_pseudo_d.close()
    fzh_orig_d.close()
    fen_d.close()

print("start copy images")
#把需要的图片
# imgae_dir_list=['/paddle/ocr_data/flow1/OCR_image_03001-04000','/paddle/ocr_data/flow2/image-filter-2',
# '/paddle/lsvt/ICDAR_lsvt_filter_images/','/paddle/icdar/ICDAR_filter_images/','/paddle/ocr_data/flow3/image_rctw/',
# '/paddle/ocr_data/flow3/image/']

# for image in image_set:
#     for image_dir in imgae_dir_list:
#         #items为当前目录下的所有图片
#         #image_list为筛选过后最后数据集残余的图片
#         items = os.listdir(image_dir)
#         items.sort()
#         image=image.strip()
#         if image in items:
#             copyfile(image_dir+'/'+image,"/paddle/whole_image_1228/")
