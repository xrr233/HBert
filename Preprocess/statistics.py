import json
all_tag_dict = {}
all_tag_num = []
with open('./statistics','r')as f:
    for line in f:
        tag_num,tag_dict = line.strip().split('\t')
        tag_dict = json.loads(tag_dict)
        for key in tag_dict:
            all_tag_dict[key] = all_tag_dict[key]+tag_dict[key] if key in all_tag_dict else tag_dict[key]
        all_tag_num.append(int(tag_num))
print(all_tag_dict)
print(all_tag_num)