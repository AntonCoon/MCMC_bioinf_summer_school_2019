import random
import collections, numpy

alphabet = sorted(list(set(list('АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя'.lower()))))+[' ']

def get_code():
    perm = numpy.random.permutation(range(len(alphabet)))
    #perm = list(range(len(alphabet)))
    print(perm)
    code = dict()
    for i in range(len(alphabet)):
        code[alphabet[i]] = alphabet[perm[i]]
        #code[' '] = ' '
    return code

def cleanstr(s):
    lstr = s.lower()
    res = ''
    for ch in lstr:
        if ch in alphabet:
            res += ch
    return res

def code_message(message, code):
    res = ''
    for letter in message:
        res += code[letter]
    return res



print(code_message('привет',get_code()))

s1 = 'Он был до того худо одет, что иной, даже и привычный человек, посовестился бы днем выходить в таких лохмотьях на улицу'
print(cleanstr(s1))
print(code_message(cleanstr(s1),get_code()))


#get_code()
#print(s1.lower())
#print(alphabet)
