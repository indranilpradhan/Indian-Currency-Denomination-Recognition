#!/usr/bin/python

import os
import cv2
from matplotlib import pyplot as plt

import subprocess
from gtts import gTTS


orbmax_val = 8
orbmax_pt = -1
orbmax_kp = 0
orbmax_des = 0

siftmax_val = 8
siftmax_pt = -1
siftmax_kp = 0
siftmax_des = 0

surfmax_val = 8
surfmax_pt = -1
surfmax_kp = 0 
surfmax_des = 0

window_name = 'matched image'
cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    test_img=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    #test_img = cv2.imread(QueryImg)
    #test_img = read_img('/home/indranil/opencv/12.jpg')
    orb = cv2.ORB_create()
    (kp1, des1) = orb.detectAndCompute(test_img, None)

    path="/home/indranil/opencv/Dataset/"
    for subdir, dirs, files in os.walk(path):
        for i in files:
	        # train image
            print(i)
            # print("hi")
            ip=os.path.join(subdir,i)
            print(ip)
            train_img=cv2.imread(ip)
        
            (kp2, des2)=orb.detectAndCompute(train_img, None)
        
	        # brute force matcher
            bf = cv2.BFMatcher()
            all_matches = bf.knnMatch(des1, des2, k=2)
            good = []
    
            for (m, n) in all_matches:
                if m.distance < 0.789 * n.distance:
                    good.append([m])

            if len(good) > orbmax_val:
                orbmax_val = len(good)
                orbmax_pt = ip
                orbmax_kp = kp2
                orbmax_des = des2

            print(i, ' ',ip, ' ', len(good))
    
    if orbmax_val != 8:
        print(orbmax_pt)
        print('good orb matches ', orbmax_val)
    '''    orb_train_img = cv2.imread(orbmax_pt)
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1,orbmax_des)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(test_img,kp1,orb_train_img,orbmax_kp,matches[:10],None, flags=2)
        (plt.imshow(img3), plt.show()) 

    else:
        print('No ORB Matches') '''
    
    sift = cv2.xfeatures2d.SIFT_create()

    (kp1, des1) = sift.detectAndCompute(test_img, None)

    for subdir, dirs, files in os.walk(path):
        for i in files:
            # train image
            print(i)
            # print("hi")
            ip=os.path.join(subdir,i)
            print(ip)
            train_img=cv2.imread(ip)
        
            (kp2, des2)=sift.detectAndCompute(train_img, None)
        
            # brute force matcher
            bf = cv2.BFMatcher()
            all_matches = bf.knnMatch(des1, des2, k=2)

            good = []
            for (m, n) in all_matches:
                if m.distance < 0.789 * n.distance:
                    good.append([m])

            if len(good) > siftmax_val:
                siftmax_val = len(good)
                siftmax_pt = ip
                siftmax_kp = kp2
                siftmax_des = des2

            print(i, ' ',ip, ' ', len(good))
    
    if siftmax_val != 8:
        print(siftmax_pt)
        print('good sift matches ', siftmax_val)
    '''   sift_train_img = cv2.imread(siftmax_pt)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(des1,siftmax_des)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(test_img,kp1,sift_train_img,siftmax_kp,matches[:10],None, flags=2)
        (plt.imshow(img3), plt.show()) 

    else:
        print('No SIFT Matches') '''
    
    surf = cv2.xfeatures2d.SURF_create()

    (kp1, des1) = surf.detectAndCompute(test_img, None)

    for subdir, dirs, files in os.walk(path):
        for i in files:
            # train image
            print(i)
            # print("hi")
            ip=os.path.join(subdir,i)
            print(ip)
            train_img=cv2.imread(ip)
        
            (kp2, des2)=surf.detectAndCompute(train_img, None)
        
            # brute force matcher
            bf = cv2.BFMatcher()
            all_matches = bf.knnMatch(des1, des2, k=2)

            good = []
 
            for (m, n) in all_matches:
                if m.distance < 0.789 * n.distance:
                    good.append([m])

            if len(good) > surfmax_val:
                surfmax_val = len(good)
                surfmax_pt = ip
                surfmax_kp = kp2
                surfmax_des = des2

            print(i, ' ',ip, ' ', len(good))
    
    if surfmax_val != 8:
        print(surfmax_pt)
        print('good surf matches ', surfmax_val)
    '''   surf_train_img = cv2.imread(surfmax_pt)
        bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=False)
        matches = bf.match(des1,surfmax_des)
        matches = sorted(matches, key = lambda x:x.distance)
        img3 = cv2.drawMatches(test_img,kp1,surf_train_img,surfmax_kp,matches[:10],None, flags=2)
        (plt.imshow(img3), plt.show())

    else:
        print('No Matches') '''
    
    if orbmax_val >=  siftmax_val and orbmax_val >= surfmax_val:
        s = orbmax_pt
        print("s ",s)
        s1 = ""
        for i in range(30, len(s)):
            if (s[i] == '/'):
                break
            s1=s1+s[i]
            i = i+1
        print(s1)
        file = "file.mp3"
        tts = gTTS(s1, 'en')
        tts.save(file)
        os.system("mpg123 " + file)
    #    image = cv2.imread(orbmax_pt)
    #    cv2.imshow(window_name,image)
    elif siftmax_val >=  orbmax_val and siftmax_val >= surfmax_val:
        s = str(siftmax_pt)
        print("s ",s)
        s1 = ""
        for i in range(30, len(s)):
            #print(s[i])
            if (s[i] == '/'):
                break
            s1=s1+s[i]
            i = i+1
        print(s1)
        file = "file.mp3"
        tts = gTTS(s1, 'en')
        tts.save(file)
        os.system("mpg123 " + file)
    #    image = cv2.imread(siftmax_pt)
    #    cv2.imshow(window_name,image)
    else:
        s = surfmax_pt
        print("s ",s)
        s1 = ""
        for i in range(30, len(s)):
            if (s[i] == '/'):
                break
            s1=s1+s[i]
            i = i+1
        print(s1)
        file = "file.mp3"
        tts = gTTS(s1, 'en')
        tts.save(file)
        os.system("mpg123 " + file)
    #    image = cv2.imread(surfmax_pt)
    #    cv2.imshow(window_name,image)
    break

cam.release()

