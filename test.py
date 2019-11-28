import nltk

s1 = ['i','always','with','baby','baby','play','!']
s2 = ['how','look','baby','play','.']

BLEUscore = nltk.translate.bleu_score.sentence_bleu([s2],s1)
print BLEUscore
