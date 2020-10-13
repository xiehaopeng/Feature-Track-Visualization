import numpy as np
import cv2


videoname = 'Deeponline9'
cap = cv2.VideoCapture(videoname+'.mp4')
# params for ShiTomasi corner detection
# maxCorners: 角点数量最大值; qualityLevel: 角点质量等级; minDistance: 两个角点间最小间距; blockSize: 计算协方差矩阵时窗口大小
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 40,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
startp = p0.copy()

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

# 计数
now_framenum = 0
now_lenth = 0
# 帧匹配间隔
frame_space = 1
# 一条轨迹总长度
track_lenth = 20

# 开始匹配
while(1):
    # 读帧计数
    ret,frame = cap.read()
    if ret is False:
            break
    now_framenum += 1
    if now_framenum % frame_space != 0:
        continue

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # good_start = startp[st==1]
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        mask = cv2.line(mask, (a,b),(c,d),(0,0,255), 2) # (0,0,255)红色；(0,255,0)绿色 这里注意在mask上画line，frame上画circle
        frame = cv2.circle(frame,(a,b),3,(0,255,0),1)
    for st in startp:
        e,f = st.ravel()
        frame = cv2.circle(frame,(e,f),3,(0,255,255),1)
    img = cv2.add(frame,mask)
    cv2.imshow('frame',img)
    now_lenth += 1

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

    # 长度足够则输出，并初始化其他参数
    if now_lenth == track_lenth:
        cv2.imwrite('result/'+videoname+'_'+str(frame_space)+'_'+str(now_framenum)+'.png', img)
        now_lenth = 0
        mask = np.zeros_like(old_frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
        startp = p0.copy()

    # 按下esc退出
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break


cv2.destroyAllWindows()
cap.release()