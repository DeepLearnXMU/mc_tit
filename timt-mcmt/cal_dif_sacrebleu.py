import sys
import nltk
import evaluate
import sacrebleu

def read_file(path):
    i = 0
    toks = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            toks.append(line)
            i += 1
    return toks, i

sys_toks, i1 = read_file(sys.argv[1])
ref_toks, i2 = read_file(sys.argv[2])

cor_toks, j1 = read_file(sys.argv[3])
ocr_toks, j2 = read_file(sys.argv[4])

assert i1 == i2, "error"
assert j1 == j2, "error"

cor_translations, cor_ref = [], []
err_translations, err_ref = [], []
ori_translations, ori_ref = [], []
cor_num, err_num = 0 , 0
for k in range(i1):
    ori_translations.append(sys_toks[k])
    ori_ref.append(ref_toks[k])
    if cor_toks[k] == ocr_toks[k]:
        cor_translations.append(sys_toks[k])
        cor_ref.append(ref_toks[k])
        cor_num+=1
    else:
        err_translations.append(sys_toks[k])
        err_ref.append(ref_toks[k])
        err_num+=1

print("err_num: {}".format(err_num))
print("cor_num: {}".format(cor_num))

# sacrebleu = evaluate.load("sacrebleu")

# ori_result = sacrebleu.compute(predictions=ori_translations,references=ori_ref)
# cor_result = sacrebleu.compute(predictions=cor_translations,references=cor_ref)
# err_result = sacrebleu.compute(predictions=err_translations,references=err_ref)


ori_result = sacrebleu.corpus_bleu(ori_translations,[ori_ref])
cor_result = sacrebleu.corpus_bleu(cor_translations,[cor_ref])
err_result = sacrebleu.corpus_bleu(err_translations,[err_ref])

# print("ori_sacrebleu: {}".format(ori_result["score"]))
# print("cor_sacrebleu: {}".format(cor_result["score"]))
# print("err_sacrebleu: {}".format(err_result["score"]))

print("ori_sacrebleu: {}".format(ori_result.score))
print("cor_sacrebleu: {}".format(cor_result.score))
print("err_sacrebleu: {}".format(err_result.score))
