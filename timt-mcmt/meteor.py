
import sys
import evaluate

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

assert i1 == i2, "error"
meteor = evaluate.load('meteor')

translations, ref = [], []
for k in range(i1):
    translations.append(sys_toks[k])
    ref.append(ref_toks[k])
meteor_score = meteor.compute(predictions=translations, references=ref)
# meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
#         translations, [ref]
#     )
print("score:")
print(meteor_score)
