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

'''
This webscraper is custom built to collect images from troutnut.com
It proceeds through the following steps:
      1. The URLs for the pages I'm interested in are all
         formatted as "http://www.troutnut.com/hatch/"+ a
         number specific to the order of insect/ + the order of
         insect/ + the page number + "#specimens"
         i.e. -
         http://www.troutnut.com/hatch/13/Insect-Plecoptera-Stoneflies
         I have the numbers and orders stored in a json file, these
         are read in.
      2. I create a deque to store the urls I'll be grabbing on my
         first pass. Troutnut has a forum setup. With the URL I created
         above, I'll be directed to the first page of that insect order's
         sub-forum. Each page of that sub-forum has 10 topics - each
         corresponding to a different specimen. I go through the
         pages for each sub-forum, and collect the URLs for the specimen
         pages, appending them to the deque.
      3. Once these have been collected, I go through the deque and visit
         each of the urls in turn. I identify the images, collect the meta
         data for each specimen, and store both image and metadata in a
         directory specific to the order of the insect.
'''

class imageScraper():

    def __init__(self):
        '''
        Calls a function to set up a few things and creates a deque
        that we'll use to store the urls for specimen pages.
        '''
        self.data_setup()
        self.Q = deque()


    def data_setup(self):
        '''
        This reads in the information stored in urlinfo.json
        We'll use this to create the urls we'll visit during our
        first pass through.
        '''
        self.data_dict = pd.read_json('urlinfo.json')
        self.insect_list = self.data_dict.orders.values
        self.tn_num = self.data_dict.tn_nums.values


    def pg_urls(self):
        '''
        Each specimen has a page which might have multiple pictures.
        This takes the url for each specimen page and appends it to
        our queue.
        '''
        p = self.html.find_all('a',attrs={'class':'vl'})
        for i in range(len(p)):
            new_url = p[i].attrs['href']
            if new_url not in self.Q:
                self.Q.append(new_url)
            else:
                break


    def page_scan(self):
        '''
        We have a list of insect orders we're looking at. The if
        statement below will trigger if it's a 'new order', i.e.
        we've just switched over from another order. If it's a new
        order, we find out how many pages there are by looking for a
        particular feature at the bottom of the page.
        '''
        req = requests.get(self.url)
        self.html = BeautifulSoup(req.content,'html.parser')
        s = self.html.find_all('div', attrs={'class':'pld'})
        if self.new_order == True:
            self.max_page = int(re.findall('[0-9]+$',s[0].get_text())[0])
            self.new_order = False
        self.pg_urls()

    def url_increment(self):
        '''
        Once we've finished with one page of the sub-forum, this function
        is called. It pauses the program for 2 seconds out of courtesy to
        troutnut, and then changes the self.url variable to the url of the
        next page
        '''
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
        '''
        This is the function that you call for a first pass through the
        website. It takes the information from the urlinfo.json file and
        uses them to build the 10 urls for the sub-forums on troutnut.
        It goes through the pages of that subforum until it reaches
        the max page, and then it moves on to the next sub-forum.
        '''
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

    def scrape(self):
        '''
        This is to be run after iter_order()
        This goes through the queue that we built in inter_order(),
        and calls grab_images for each page until the queue is empty.
        '''
        self.unpickle_queue()
        print('Image queue loaded...')
        while len(self.Q) > 0:
            print('Grabbing images...')
            self.grab_images()
            time.sleep(3)

    def grab_images(self):
        '''
        This grabs the images and metadata from each page and for
        each specimen, and saves both the images and the
        metadata to a directory specified in urlinfo.json

        '''
        self.url = self.Q.popleft()
        req = requests.get(self.url)
        html = BeautifulSoup(req.content, 'html.parser')
        a = html.find_all('img', attrs={'class':'i'})
        order_url = html.find_all('a', attrs={'itemprop':'url'})[3]['href']
        order_val = order_url.split('/')[5]
        t = html.find_all('span', attrs = {'itemprop':'title'})
        order_dir = self.data_dict[self.data_dict.orders == order_val].directory.values[0]
        print('Found images in',order_url)
        meta_info = []
        for i in range(len(a)):
            b = a[i]
            src = b.attrs['src']
            sub_source = src.split('/')[4]
            # There were images from 'im_glossary' that appeared on
            # several different images - made things tricky
            if sub_source != 'im_regspec':
                c = [b.attrs['name'],b.attrs['title'],b.attrs['alt'],src]
                taxo = [x.get_text() for x in t[3:]]
                d = ';'.join(c+taxo+['\n'])
                meta_info.append(d)
                img_arr = imread(requests.get(src, stream=True).raw)
                imsave("../data/troutnut/{}/{}.jpg".format(order_dir,a[i]['name']), img_arr)
        print(len(a), "images saved successfully")
        with open("../data/troutnut/{}/meta.txt".format(order_dir),"a") as f:
            for i in range(len(meta_info)):
                #f.write(image_name[i]+','+source_info[i]+','+source_info[i]+','+source_info_alt+','+image_url[i])
                f.write(meta_info[i])
        print('Updated metadata.')
        self.pickle_queue()
        print('Updated queue')

    def repopulate_queue(self):
        '''
        Don't pay too much attention to this - this is just
        a recovery option for if something gets messed up. I was
        tinkering with the code, fixing bugs, making improvements
        as I went along.
        It rebuilds the queue for a specific insect order, that way
        I don't have to run the entire thing a second time

        I debated removing these blocks of code, but decided not to.
        Removed them from the other web scraper, but left them in place
        here just to show the messy, messy process to get here.
        '''
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
        '''
        pickles the queue so that I can exit the code and come
        back to it, picking up where I left off.
        useful for tinkering.
        '''
        with open('pickle/Q.pkl','wb') as f:
            pickle.dump(self.Q,f)

    def unpickle_queue(self):
        '''
        recovers the previously pickled queue
        '''
        with open('pickle/Q.pkl','rb') as f:
            self.Q = pickle.load(f)

    def pickle_queue_master(self):
        # backup of the queue
        with open('pickle/Q_master.pkl','wb') as f:
            pickle.dump(self.Q,f)
        self.pickle_queue()

    def unpickle_master(self):
        # unpickle backup
        with open('pickle/Q_master.pkl','rb') as f:
            self.Q = pickle.load(f)
        self.pickle_queue()




















# #
