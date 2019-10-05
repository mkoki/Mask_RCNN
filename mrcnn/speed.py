#元の動画の情報を見て、1920 ×　1080　のようにサイズを合わせる
w = 768
h = 432

#動画の半分のスピードになります。　　３０にすれば1倍の速さ。　ex. 10 =  3分の１の速さ
speed = 30

#movie_nameにGoogle Driveの動画名を入れる
movie_name = 'fede.MOV'
!cp "./gdrive/My Drive/fede.MOV" "./"

import cv2

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video  = cv2.VideoWriter('output.mp4', fourcc, speed, (w, h))

if __name__ == '__main__':

    outimg_files = []
    count = 0
    cap = cv2.VideoCapture(movie_name)
    ball_list = []
    col_list = []
    x_ma_list = []
    x_mi_list = []

    while True:
        ret, image = cap.read()

        if ret == True:
            # １フレームずつ処理
            count += 20
            image = cv2.resize(image,(w,h))
            results = model.detect([image], verbose=0)
            r = results[0]
            for i, item in enumerate(r['class_ids']):
              if item == 33:
                ball = r['masks'][:,:,i].astype(np.int) * 255
                ball_pixel = np.where(ball == 255)
                x_min = np.amin(ball_pixel[1])
                x_max = np.amax(ball_pixel[1])
                index = np.average(ball_pixel[0])
                col = np.average(ball_pixel[1])
                col_list.append(col)
                x_ma_list.append(x_max)
                x_mi_list.append(x_min)

        else:
            break
    video.release()
np.amax(np.diff(np.array(col_list)))
x_max = np.average(np.array(x_ma_list))
x_min = np.average(np.array(x_mi_list))
print(round(np.amax(np.diff(np.array(col_list))) / (x_max - x_min) * 6.7 / 100000 * 120 * 3600, 1), 'km/h')
