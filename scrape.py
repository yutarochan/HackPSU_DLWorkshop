'''
Dataset Scrape & Preprocessor
Author: Yuya Jeremy Ong (yuyajeremyong@gmail.com)
'''
import os
import sys
import cv2
import dlib
from os import listdir
from skimage import io
from os.path import isfile, join
from multiprocessing import Pool
from icrawler.builtin import GoogleImageCrawler

def processImage(image_path):
    try:
        print 'PROCESSING: ' + image_path
        image = cv2.cvtColor(io.imread(image_path), cv2.COLOR_BGR2RGB)
        detector = dlib.get_frontal_face_detector()
        dets = detector(image, 1)

        for i, d in enumerate(dets):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(i, d.left(), d.top(), d.right(), d.bottom()))
            crop = image[d.top():d.bottom(),d.left():d.right()]
            cv2.imwrite(output_dir_pro+'/'+image_path.split('/')[-1].split('.')[0]+'-'+str(i)+'.jpg', crop)
    except Exception as e:
        return

if len(sys.argv) < 3:
    print 'Usage: python scrape.py \'<query>\' <count>'
    sys.exit()

keyword = sys.argv[1]
count = int(sys.argv[2])

output_dir_raw = 'data/raw/'+keyword
output_dir_pro = 'data/proc/'+keyword

if not os.path.exists(output_dir_raw): os.makedirs(output_dir_raw)
if not os.path.exists(output_dir_pro): os.makedirs(output_dir_pro)

print '[Crawling Images]: ' + keyword
google_crawler = GoogleImageCrawler(parser_threads=2, downloader_threads=4, storage={'root_dir': output_dir_raw})
google_crawler.crawl(keyword=keyword, max_num=count, date_min=None, date_max=None, min_size=(200,200), max_size=None)

print '\n[Preprocessing Data]'

image_files = [output_dir_raw+'/'+f for f in listdir(output_dir_raw) if isfile(join(output_dir_raw, f))]

p = Pool(8)
p.map(processImage, image_files)
p.close()

print '\n>> [Complete]'
