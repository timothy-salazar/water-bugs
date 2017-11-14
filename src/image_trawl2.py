import pandas as pd
import numpy as np
import string
from bs4 import BeautifulSoup
from collections import deque
import requests
import re
import time
import pickle
from skimage.io import imread, imsave

class imageScraper():

    def __init__(self):
        #self.Q = queue.Queue()
        self.data_setup()
        self.Q = deque()

    def data_setup(self):
        self.data_dict = pd.read_json('urlinfo.json')
        self.insect_list = self.data_dict.orders.values
        self.tn_num = self.data_dict.tn_nums.values

    def pg_urls(self):
        p = self.html.find_all('a',attrs={'class':'vl'})
        for i in range(len(p)):
            new_url = p[i].attrs['href']
            if new_url not in self.Q:
                self.Q.append(new_url)
            else:
                break

    def page_scan(self):
        req = requests.get(self.url)
        self.html = BeautifulSoup(req.content,'html.parser')
        s = self.html.find_all('div', attrs={'class':'pld'})
        if self.new_order == True:
            self.max_page = int(re.findall('[0-9]+$',s[0].get_text())[0])
            self.new_order = False
        self.pg_urls()

    def url_increment(self):
        # note - this is brilliant, but only if the url ends with '/'
        time.sleep(2)
        old_end = re.findall('[a-z0-9#]+$',self.url)
        print('old end',old_end)
        if len(old_end) > 0:
            old_end = old_end[0]
            new_num = str(int(re.findall('[0-9]+',old_end)[0]) + 1)
            self.url = self.url.replace(old_end,'{}#specimens'.format(new_num))
        else:
            self.url = self.url + '2#specimens'
        print('url:', self.url)

    def iter_order(self):
        for (num,ins) in zip(self.tn_num,self.insect_list):
            print('Beginning:',ins)
            self.url = "http://www.troutnut.com/hatch/{}/{}/".format(num,ins)
            self.new_order = True
            self.page_scan()
            for pg in range(self.max_page):
                print('page {} of {}'.format(pg,self.max_page))
                self.url_increment()
                self.page_scan()
        self.pickle_queue_master()

    def repopulate_queue(self):
        self.unpickle_queue()
        number = input("num -->")
        print('Beginning...')
        row = self.data_dict.loc[int(number)]
        num = row[2]
        ins = row[1]
        self.url = "http://www.troutnut.com/hatch/{}/{}/".format(num,ins)
        self.new_order = True
        self.page_scan()
        print('setup complete, entering loop...')
        for pg in range(self.max_page):
            print('page {} of {}'.format(pg,self.max_page))
            self.url_increment()
            self.page_scan()
        print('pickling queue...')
        self.pickle_queue_master()
        print('complete')

    def pickle_queue(self):
        with open('Q.pkl','wb') as f:
            pickle.dump(self.Q,f)

    def unpickle_master(self):
        with open('Q_master.pkl','rb') as f:
            self.Q = pickle.load(f)
        self.pickle_queue()

    def pickle_queue_master(self):
        with open('Q_master.pkl','wb') as f:
            pickle.dump(self.Q,f)
        self.pickle_queue()

    def unpickle_queue(self):
        with open('Q.pkl','rb') as f:
            self.Q = pickle.load(f)
        # self.tn_num = self.tn_num[7:]
        # self.insect_list = self.insect_list[7:]

    def scrape(self):
        self.unpickle_queue()
        print('Image queue loaded...')
        while len(self.Q) > 0:
            print('Grabbing images...')
            self.grab_images()
            time.sleep(3)

    def grab_images(self):
        self.url = self.Q.popleft()
        req = requests.get(self.url)
        html = BeautifulSoup(req.content, 'html.parser')
        a = html.find_all('img', attrs={'class':'i'})
        order_url = html.find_all('a', attrs={'itemprop':'url'})[3]['href']
        order_val = order_url.split('/')[5]
        t = html.find_all('span', attrs = {'itemprop':'title'})
        order_dir = self.data_dict[self.data_dict.orders == order_val].directory.values[0]
        print('Found images in',order_url)
        # image_name = []
        # source_info = []
        # source_info_alt = []
        # image_url = []
        meta_info = []
        for i in range(len(a)):
            b = a[i]
            src = b.attrs['src']
            c = [b.attrs['name'],b.attrs['title'],b.attrs['alt'],src]
            taxo = [x.get_text() for x in t[3:]]
            d = ';'.join(c+taxo+['\n'])
            meta_info.append(d)
            img_arr = imread(requests.get(src, stream=True).raw)
            imsave("../images/troutnut/{}/{}.jpg".format(order_dir,a[i]['name']), img_arr)
        print(len(a), "images saved successfully")
        with open("../images/troutnut/{}/meta.txt".format(order_dir),"a") as f:
            for i in range(len(meta_info)):
                #f.write(image_name[i]+','+source_info[i]+','+source_info[i]+','+source_info_alt+','+image_url[i])
                f.write(meta_info[i])
        print('Updated metadata.')
        self.pickle_queue()
        print('Updated queue')


#
# if __name__ == '__main__':
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# #
