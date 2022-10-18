import os 
import json
import pandas as pd


js_path = '/dataset/train/instances_train.json'
def generate_csv(js_path=js_path, csv_path ="./folds.csv", fold_num = 5):
    coco2 = json.load(open(js_path,'r'))
    
    lnth = len(coco2["images"])
    fold_len = lnth//fold_num

    names = []
    fold = []
    ids = []
    for i, im in enumerate(coco2["images"]):
        f = i // fold_len
        names.append(im["file_name"])
        fold.append(f)
        ids.append(im['id'])

    with open(csv_path, 'w') as f:
        f.write('id,name,fold\n')
        for id, name, fold in zip(ids, names, fold):
            f.write(str(id)+','+name+','+str(fold)+'\n')

if __name__=="__main__":
    generate_csv(js_path=js_path, csv_path ="./folds.csv", fold_num = 5)
    df = pd.read_csv('./folds.csv',)
    
    img_ids = [1,100, 999]
    img_valid = []
    folds = {0,1,2,3}
    for img_id in img_ids:
        f = df[df['id']==img_id]['fold'].tolist()[0]
        if f in folds:
            img_valid.append(img_id)
