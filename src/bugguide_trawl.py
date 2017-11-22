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

# This webscraper is custom built to collect images from troutnut.com
# It proceeds through the following steps:
#       1. The URLs for the pages I'm interested in are all
#          formatted as "http://www.troutnut.com/hatch/"+ a
#          number specific to the order of insect/ + the order of
#          insect/ + the page number + "#specimens"
#          i.e. -
#          http://www.troutnut.com/hatch/13/Insect-Plecoptera-Stoneflies
#          I have the numbers and orders stored in a json file, these
#          are read in.
#       2. I create a deque to store the urls I'll be grabbing on my
#          first pass. Troutnut has a forum setup. With the URL I created
#          above, I'll be directed to the first page of that insect order's
#          sub-forum. Each page of that sub-forum has 10 topics - each
#          corresponding to a different specimen. I go through the
#          pages for each sub-forum, and collect the URLs for the specimen
#          pages, appending them to the deque.
#       3. Once these have been collected, I go through the deque and visit
#          each of the urls in turn. I identify the images, collect the meta
#          data for each specimen, and store both image and metadata in a
#          directory specific to the order of the insect.

class imageScraper():

    def __init__(self):
        # read in the url data we'll need later (the formatting of
        # the url for each order we're interested in), and create a
        # deque for storing the urls for specimen pages.
        #self.data_setup()
        self.Q = deque()
        self.imgQ = deque()
        self.u_start = 'https://bugguide.net/'

    # This appends the urls for the specimen pages to our queue.
    def pg_urls(self):
        p = self.html.find_all('a', attrs={'class':'bb_url'})
        for i in p:
            new_url = i.attrs['href']
            if new_url not in self.Q:
                self.Q.append(new_url)
            else:
                pass

    # this looks at the current page, finds the highest page in the
    # forum from a navigation bar at the bottom, and calls pg_urls
    def page_scan(self):
        req = requests.get(self.url)
        self.html = BeautifulSoup(req.content,'html.parser')
        self.pg_urls()

    # This is the function that you call for a first pass through the
    # website. It takes the information from the urlinfo.json file and
    # uses them to build the 10 urls for the sub-forums on troutnut.
    # It goes through the pages of that subforum until it reaches
    # the max page, and then it moves on to the next sub-forum.
    def iter_order(self):
        u_list = ['https://bugguide.net/node/view/5233',
                'https://bugguide.net/node/view/76',
                'https://bugguide.net/node/view/55',
                'https://bugguide.net/node/view/78']
        file_paths = ['../data/bug_guide/trichoptera','../data/bug_guide/plecoptera',
        '../data/bug_guide/diptera','../data/bug_guide/ephemeroptera']
        for u, f in zip(u_list, file_paths):
            self.url = u
            self.file_path = f
            print('Beginning:',self.url)
            self.page_scan()
            self.scrape()

    # This is to be run after iter_order()
    # This goes through the queue that we built in inter_order(),
    # grabs the images from each page, grabs the metadata from each
    # page and for each specimen, and saves both the images and the
    # metadata to a directory specified in urlinfo.json

    def scrape(self):
        print('Image queue loaded...')
        while len(self.Q) > 0:
            print('Grabbing images...')
            self.grab_source()
            self.grab_images()
            time.sleep(3)

    def grab_source(self):
        self.url = self.u_start + self.Q.popleft()
        req = requests.get(self.url)
        html = BeautifulSoup(req.content, 'html.parser')
        a = html.find_all('img', attrs={'class':'bgimage-thumb'})
        for i in a:
            if i not in self.imgQ:
                self.imgQ.append(i.parent.attrs['href'])

    # this is misleadingly named - it grabs the images and the metadata
    # scrape will call this function until the queue is empty.
    def grab_images(self):
        while len(self.imgQ) > 0:
            time.sleep(2)
            img_url = self.imgQ.popleft()
            print('Fetching ',img_url)
            req = requests.get(img_url)
            self.html = BeautifulSoup(req.content,'html.parser')
            a = self.html.find('img',attrs={'class':'bgimage-image'})
            src = a.attrs['src']
            img_arr = imread(requests.get(src, stream=True).raw)
            self.image_id = self.html.find('td',attrs={'class':'bgimage-id'}).get_text()
            self.image_id = self.image_id.lower().replace('#','_') + '.jpg'
            img_fp = self.file_path + '/' + self.image_id
            imsave(img_fp,img_arr)
            self.get_meta(img_arr)

    def get_meta(self,img_arr):
            print('Collecting metadata')
            b = self.html.find('div',attrs={'class':'bgimage-where-when'})
            c = str(b).split('<br/>')
            location = c[0][32:]
            try:
                date_collected = c[1].replace('</div>','None')
            except:
                date_collected = 'None'
            try:
                size = c[2].replace('</div>','None')
            except:
                size = 'None'
            taxon = self.html.find('div', attrs={'class':'bgpage-roots'})
            taxon = taxon.find_all('a')
            self.tax_dict = dict()
            for i in range(4,len(taxon)):
                t = taxon[i].get_text().replace('\xa0',' ')
                self.tax_dict[taxon[i].attrs['title']] = t
            taxon_list = []
            t5='\n'
            for i in ['Order','Family','Genus','Species']:
                try:
                    print(self.tax_dict[i])
                    taxon_list.append(self.tax_dict[i])
                except:
                    taxon_list.append('None')
            self.ml = [self.image_id, location, date_collected,
                        size, taxon_list[0], taxon_list[1], taxon_list[2],
                        taxon_list[3], t5]
            with open(self.file_path + '/meta.txt',"a") as f:
                f.write(';'.join(self.ml))
            print('Metadata saved:',self.ml)


#file_name;location;date_collected;size,order;family;genus;species;

        # self.url = self.Q.popleft()
        # req = requests.get(self.url)
        # html = BeautifulSoup(req.content, 'html.parser')
        # a = html.find_all('img', attrs={'class':'bgimage-thumb'})
        # order_url = html.find_all('a', attrs={'itemprop':'url'})[3]['href']
        # order_val = order_url.split('/')[5]
        # t = html.find_all('span', attrs = {'itemprop':'title'})
        # order_dir = self.data_dict[self.data_dict.orders == order_val].directory.values[0]
        # print('Found images in',order_url)
        # meta_info = []
        # for i in range(len(a)):
        #     b = a[i]
        #     src = b.attrs['src']
        #     c = [b.attrs['name'],b.attrs['title'],b.attrs['alt'],src]
        #     taxo = [x.get_text() for x in t[3:]]
        #     d = ';'.join(c+taxo+['\n'])
        #     meta_info.append(d)
        #     img_arr = imread(requests.get(src, stream=True).raw)
        #     imsave("../data/troutnut/{}/{}.jpg".format(order_dir,a[i]['name']), img_arr)
        # print(len(a), "images saved successfully")
        # with open("../data/troutnut/{}/meta.txt".format(order_dir),"a") as f:
        #     for i in range(len(meta_info)):
        #         #f.write(image_name[i]+','+source_info[i]+','+source_info[i]+','+source_info_alt+','+image_url[i])
        #         f.write(meta_info[i])
        # print('Updated metadata.')
        # print('Updated queue')
















# #
