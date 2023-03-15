import cv2
import json
str='res-demo/res-4'
src=cv2.imread(str+'.png')
with open(str+'.json','r') as f:
    js=json.load(f)
    for i in js['shapes']:
        print(i)
        for j in i['points']:
            cv2.circle(src,(int(j[0]),int(j[1])),1,(0,255,0),-1)

cv2.namedWindow('a',cv2.WINDOW_NORMAL+cv2.WINDOW_KEEPRATIO)
cv2.imshow('a',src)
cv2.waitKey(0)