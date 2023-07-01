from itertools import zip_longest
from PIL import Image
import os
import judge_chinese
import numpy as np
from skimage.draw import polygon
import shapely
from shapely.geometry import Polygon, MultiPoint  # 多边形

# 该文件用于ocr识别结束后，将正确文本加入到没匹配上的框中


# Paddleocr目前支持中英文、英文、法语、德语、韩语、日语，可以通过修改lang参数进行切换
# 参数依次为`ch`, `en`, `french`, `german`, `korean`, `japan`。
# 显示结果
def polygon_IOU(polygon_1, polygon_2):
    """
    计算两个多边形的IOU
    :param polygon_1: [[row1, col1], [row2, col2], ...]
    :param polygon_2: 同上
    :return:
    """
    rr1, cc1 = polygon(polygon_2[:, 0], polygon_2[:, 1])
    rr2, cc2 = polygon(polygon_1[:, 0], polygon_1[:, 1])

    try:
        r_max = max(rr1.max(), rr2.max()) + 1
        c_max = max(cc1.max(), cc2.max()) + 1
    except:
        return 0

    canvas = np.zeros((r_max, c_max))
    canvas[rr1, cc1] += 1
    canvas[rr2, cc2] += 1
    union = np.sum(canvas > 0)
    if union == 0:
        return 0
    intersection = np.sum(canvas == 2)
    return intersection / union

def compute_IOU(rec1,rec2):
    """
    计算两个矩形框的交并比。
    :param rec1: (x0,y0,x1,y1)      (x0,y0)代表矩形左上的顶点，（x1,y1）代表矩形右下的顶点。下同。
    :param rec2: (x0,y0,x1,y1)
    :return: 交并比IOU.
    """
    left_column_max  = max(rec1[0],rec2[0])
    right_column_min = min(rec1[2],rec2[2])
    up_row_max       = max(rec1[1],rec2[1])
    down_row_min     = min(rec1[3],rec2[3])
    #两矩形无相交区域的情况
    if left_column_max>=right_column_min or down_row_min<=up_row_max:
        return 0
    # 两矩形有相交区域的情况
    else:
        S1 = (rec1[2]-rec1[0])*(rec1[3]-rec1[1])
        S2 = (rec2[2]-rec2[0])*(rec2[3]-rec2[1])
        S_cross = (down_row_min-up_row_max)*(right_column_min-left_column_max)
        return S_cross/(S1+S2-S_cross)
#测试样例1
r1=[639,34,715,50]
r2=[637,36,716,52]
IOU = compute_IOU(r1,r2)
print('rectangle_iou:',IOU)

def isnum(num):
    try:
        s=int(num)
        return True
    except ValueError as e:
        return False
def find_position_end_index(sent):
    position_end_index=0
    for s in sent:
        if not isnum(s):
            break
        position_end_index+=1
    if position_end_index%2!=0:
        position_end_index-=1
    return position_end_index
# IOU=calulate_iou(r1,r2)
# print('polygon_iou:',IOU)
# ocr = PaddleOCR(use_angle_cls=True,use_gpu=False,det_model_dir='ch_ppocr_server_v2.0_det_infer',
# cls_model_dir='ch_ppocr_mobile_v2.0_cls_infer',rec_model_dir='ch_ppocr_server_v2.0_rec_infer',
# lang="ch")  # need to run only once to download and load model into memory

# ocr = PaddleOCR(use_angle_cls=True,use_gpu=True,
# lang="ch")  # need to run only once to download and load model into memory
min_iou=0.5
dataset='lsvt'
poly_flag=False #是否为多边形

data_dir='../v2.0/'
labeled_text_path=data_dir+'all_labeled_v2/'
img_par_path = data_dir+'whole_image_v2/'
orig_text_path = data_dir+'raw_data/'
tgt_text_path=data_dir+'ocr_pseudo_cor_mix/'
in_text_path=data_dir+'ocr_pseudo_all_'+str(min_iou)+'/'

if not os.path.exists(tgt_text_path):
    os.makedirs(tgt_text_path)

orig_text_p = os.listdir(labeled_text_path)
orig_text_p.sort()

# print('len items:',len(items))
print('len orig_text_p:',len(orig_text_p))

# map(None, items,orig_text_p)
# print('after map len items:',len(items))
# print('after map len orig_text_p:',len(orig_text_p))
# tgt_text_path='/paddle/ocr_data/flow3/ocr_pseudo_pal/'
# iter_list=zip_longest(items,orig_text_p)/
cur_num=0
f_vaild=open(data_dir+'after_proprecess/after_tok_bpe/image_valid.txt','r',encoding='UTF-8')
f_test=open(data_dir+'after_proprecess/after_tok_bpe/image_test.txt','r',encoding='UTF-8')
vaild_images = f_vaild.readlines()
test_images=f_test.readlines()
for orig_text in orig_text_p:

    # img_path=orig_text[:-4]+'.jpg'

    # img_path='PAL10175.jpg'
    # orig_text='PAL10175.txt'

    #result = ocr.ocr(img_par_path+img_path, cls=True)

    # if img_path==None or orig_text==None:
    #     continue
    if orig_text==None:
        continue
    if not os.path.exists(in_text_path+orig_text):
        continue
    cur_num+=1
    print("cur_num",cur_num)
    f_pseudo=open(in_text_path+orig_text,'r',encoding='UTF-8')
    fW = open(tgt_text_path+orig_text, 'w', encoding='UTF-8')
    try:
        fR = open(orig_text_path+orig_text, 'r', encoding='UTF-8')
        sentences = fR.readlines()
    except:
        try:
            fR = open(orig_text_path+orig_text, 'r', encoding='gbk')
            sentences = fR.readlines()
        except:
            print(orig_text)
            continue
    origin_pseudo=f_pseudo.readlines()
    #对于验证集测试集，按OCR和正确文本匹配的来
    img_path=orig_text[:-4]+'.jpg\n'
    if img_path in vaild_images or img_path in test_images:
        print('in test')
    else:
        for i in range(len(origin_pseudo)):
            if origin_pseudo[i]=='*\n':
                sent=sentences[i].split(',')
                position_end_index=find_position_end_index(sent)
                if 'tr_img_' in orig_text:
                    position_end_index+=1
                str_a=sent[position_end_index:]
                orig_text_o=','.join(str_a)
                origin_pseudo[i]=orig_text_o.strip()+'||'+orig_text_o
    for ins in range(len(sentences)):
        r=origin_pseudo[ins].strip()
        fW.write("%s\n" % r)
    fR.close()
    fW.close()