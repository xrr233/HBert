import linecache
import json
import random

rangf_path = 'src/range_0.jsonl'
src_path = 'src/bigfile_0'
web_path = 'web/'

lis = []
for i in range(5):
    x = random.randint(0, 10000)
    lis.append(x)
lis.sort()
Bylis = []
namelis = []

for i in range(5):
    line = linecache.getline(rangf_path, lis[i])
    data = json.loads(line)
    Bylis.append(data['range'])
    namelis.append(data['id'])


with open(src_path,"rb") as f:

    for i in range(5):
        with open(web_path + '{}.html'.format(namelis[i]), 'wb') as webf:
            st = Bylis[i]
            poi = st.find('-')
            w_begin = int(st[:poi])
            w_end = int(st[poi+1:]) + 1 #close-open
            f.seek(w_begin)
            data = f.read(w_end - w_begin)
            webf.write(data)
        
