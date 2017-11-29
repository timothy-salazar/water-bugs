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

# This webscraper is custom built to collect images from bugguide.net
# It proceeds through the following steps:
#       1. I grab a url and the path to a directory from a list. The url
#          corresponds to an order of insect, and the path is the folder
#          where I want to save the images and metainformation for that order.
#       2. I visit the first url, find the links that I'm interested in (the
#          ones that lead to family, genus, or species pages for the insect 
#          order we're working on), and appends them to a queue.
#       3. Once these have been collected, I go through the deque and visit
#          each of the urls in turn. I identify the images, collect the meta
#          data for each specimen, and store both image and metadata in a
#          directory specific to the order of the insect.
#       4. This is repeated for the next url and directory in the list.

class imageScraper():

    def __init__(self):
    # read in the url data we'll need later (the formatting of
    # the url for each order we're interested in), and create a
    # deque for storing the urls for specimen pages.
    #self.data_setup()
        self.Q = deque()
        self.imgQ = deque()
        self.u_start = 'https://bugguide.net/'


    # This is the function we call during our first pass through the
    # website. It goes through u_list - which contains the urls for the
    # pages detailing the orders we're interested in. Each of these pages
    # contain urls leading to more pages (for family, genus, species, etc.),
    # and we add these to our queue with page_scan(). We'll go through these
    # and collect more urls - and then images - by calling scrape().


    #It takes the information from the urlinfo.json file and
    # uses them to build the 10 urls for the sub-forums on troutnut.
    # It goes through the pages of that subforum until it reaches
    # the max page, and then it moves on to the next sub-forum.
    def iter_order(self):
        u_list = ['https://bugguide.net/node/view/78']
        file_paths = ['../data/bug_guide/ephemeroptera']
        for u, f in zip(u_list, file_paths):
            self.url = u
            self.file_path = f
            print('Beginning:',self.url)
            self.page_scan()
            self.scrape()

    # this creates a BeautifulSoup object and then calls pg_urls, which
    # will go through it and append the urls we're interested in to our queue.
    def page_scan(self):
        req = requests.get(self.url)
        self.html = BeautifulSoup(req.content,'html.parser')
        self.pg_urls()

    def pg_urls(self):
        p = self.html.find_all('a', attrs={'class':'bb_url'})
        for i in p:
            new_url = i.attrs['href']
            if new_url not in self.Q:
                self.Q.append(new_url)
            else:
                pass

    # This function will call grab_source(), which collects the urls for
    # the actual image files and appends them to a separate queue, and
    # grab_images(), which saves the images.

    def scrape(self):
        print('Image queue loaded...')
        while len(self.Q) > 0:
            print('Grabbing images...')
            self.grab_source()
            self.grab_images()

    # For each of the urls attached to the queue in pg_urls(), this
    # function collects the urls of the image files and appends them
    # to a second queue called imgQ

    def grab_source(self):
        self.url = self.u_start + self.Q.popleft()
        req = requests.get(self.url)
        html = BeautifulSoup(req.content, 'html.parser')
        a = html.find_all('img', attrs={'class':'bgimage-thumb'})
        for i in a:
            if i not in self.imgQ:
                self.imgQ.append(i.parent.attrs['href'])

    # For each of the urls attached to imgQ in grab_source(), this
    # function saves the image to a directory specific to the order of
    # insect it represents, and then calls get_meta(), which will write it to
    # a meta information file

    def grab_images(self):
        while len(self.imgQ) > 0:
            time.sleep(3)
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

    # We saved the BeutifulSoup object from grab_images() as self.html, so
    # it's still accessible to this function. Here we're performign a few
    # searches on it to pull useful information from the page we collected
    # the image from:
    # <div>class=bgimage-where-when  <- contains geographic/time information
    # about the specimen.
    # We also pull taxonomic information from the page, with as much
    # resolution as is available.
    # This is formatted and appended to a metadata file.

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








#
