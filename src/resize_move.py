import numpy as np
import pandas as pd
from skimage.io import imread, imsave, imshow
from skimage.transform import resize, rescale
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

df = pd.read_csv('../data/processed/meta.txt',sep=';')
df = df.append(pd.read_csv('../data/processed_2/meta.txt',sep=';'))
df['index'] = np.arange(df.shape[0])
df.set_index('index', inplace=True)

dir_list = ['plecoptera','trichoptera','diptera','ephemeroptera']
bug_columns = ['file_name','location','date_collected',
                'size','order','family','genus','species']
bug_df = pd.DataFrame()
trout_columns = ['file_name','title','alt_title','image_source','order'
                'family','genus','species','extra_column']
trout_df = pd.DataFrame()

for d in dir_list:
    bug_path = '../data/bug_guide/{}/meta.txt'.format(d)
    trout_path ='../data/troutnut/{}/meta.txt'.format(d)
    bug_df = bug_df.append(pd.read_csv(bug_path,sep=';'))
    trout_df = trout_df.append(pd.read_csv(trout_path,sep=';'))

# bug_df['index'] = np.arange(bug_df.shape[0])
# bug_df.set_index('index', inplace=True)
# trout_df['index'] = np.arange(trout_df.shape[0])
trout_df.file_name = trout_df.file_name +'.jpg'
# trout_df.set_index('index', inplace=True)
df['file_path'] = df.file_name
df['file_name'] = [i.split('/')[4] for i in df.file_name]
df_cols = ['file_name','file_path','ready','back_view','side_view','ruler','hand_nature',
            'multiple','contrast','noisy_background', 'other']
df = df[df_cols]

columns = ['file_name','order','family']
trout_df = trout_df[columns]
bug_df = bug_df[columns]
temp_df = pd.concat([bug_df,trout_df])
meld_df = pd.merge(df,temp_df,how='inner',on='file_name')
a = np.unique(meld_df.file_name,return_counts = True)
b = [np.where(meld_df.file_name == i)[0] for i in a[0][a[1]>1]]
for i in b:
    meld_df.drop(i[:-1],inplace=True)



# bug_where = []
# bug_img = []
# trout_where = []
# trout_img = []
#
# for i in df.short_name:
#     j = np.where(bug_df.file_name == i)[0]
#     if j.shape[0] > 0:
#         bug_where.append(j[0])
#         bug_img.append(df.loc[j[0]][img_cols].values)
#     j = np.where(trout_df.file_name == i)[0]
#     if j.shape[0] > 0:
#         trout_where.append(j[0])
#         trout_img.append(df.loc[j[0]][img_cols].values)
# # bug_where = np.array(bug_where)
# # trout_where = np.array(trout_where)
#
# columns = ['file_name','order','family']
# img_cols = ['file_path','ready','back_view','side_view','ruler','hand_nature',
#             'multiple','contrast','noisy_background', 'other']
#
# bug_img = pd.DataFrame(bug_img)
# bug_img.columns = img_cols
# bug_img['index'] = np.arange(bug_img.shape[0])
# bug_img.set_index('index', inplace=True)
#
# bug_df = bug_df.loc[bug_where][columns]
# bug_df['index'] = np.arange(bug_df.shape[0])
# bug_df.set_index('index',inplace=True)
#
# trout_img = pd.DataFrame(trout_img)
# trout_img.columns = img_cols
# trout_img['index'] = np.arange(trout_img.shape[0])
# trout_img.set_index('index', inplace=True)
#
# trout_df = trout_df.loc[trout_where][columns]
# trout_df['index'] = np.arange(trout_df.shape[0])
# trout_df.set_index('index',inplace=True)
#
# meld_bug = pd.concat([bug_df,bug_img],axis=1)
# meld_trout = pd.concat([trout_df,trout_img],axis=1)
# meld_df = meld_bug.append(meld_trout)
#
# meld_df['index'] = np.arange(meld_df.shape[0])
# meld_df.set_index('index',inplace=True)
# #

order_dict = {'Stoneflies (Plecoptera)':'Plecoptera','Plecoptera (Stoneflies)':'Plecoptera',
'Mayflies (Ephemeroptera)':'Ephemeroptera','Ephemeroptera (Mayflies)':'Ephemeroptera',
'Caddisflies (Trichoptera)':'Trichoptera','Trichoptera (Caddisflies)':'Trichoptera',
'Diptera (True Flies)':'Diptera','Flies (Diptera)':'Diptera'}
order_list = []
for i in meld_df.order:
    order_list.append(order_dict[i])
meld_df.order = order_list


for i in meld_df.file_path:

    a = i.split('/')
    d = '../data/resized/{}'.format(a[4])
    del a[3]
    a = '/'.join(a).replace('bug_guide','processed').replace('troutnut','processed_2')
    b = imread(a)
    c = resize(b,(299,299))
    imsave(d,c)
with open('../data/resized/meta.txt','w') as f:
    for i in range(meld_df.shape[0]):
        f.write(';'.join(meld_df.iloc[i].values.astype('str'))+'\n')















#
