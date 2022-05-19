import common as cm
import cv2
import numpy as np
from PIL import Image
import time
from threading import Thread
import YB_Pcb_Car 
import threading
import sys
import RPi.GPIO as GPIO
import EmailSending as email

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter.fourcc('M', 'J', 'P', 'G'))
cap.set(cv2.CAP_PROP_BRIGHTNESS, 60) 
cap.set(cv2.CAP_PROP_CONTRAST, 50)
cap.set(cv2.CAP_PROP_EXPOSURE, 156)
threshold=0.5
top_k=5

model_dir = '/home/pi/fyp/Functions/all_models'
model = 'mobilenet_ssd_v2_coco_quant_postprocess.tflite'
model_edgetpu = 'mobilenet_ssd_v2_coco_quant_postprocess_edgetpu.tflite'
lbl = 'coco_labels.txt'

previousTime = 0
count = 0

tolerance=0.1
x_deviation=0
y_max=0
car = YB_Pcb_Car.YB_Pcb_Car()
car.Ctrl_Servo(1,85)
car.Ctrl_Servo(2,85)
FallState = "Stay"
scale = 0

arr_track_data=[0,0,0,0,0,0]

object_to_track='person'
#------------GPIO init------------------
GPIO.setmode(GPIO.BOARD)

GPIO.setwarnings(False)
EchoPin = 18
TrigPin = 16

GPIO.setup(EchoPin,GPIO.IN)
GPIO.setup(TrigPin,GPIO.OUT)

#---------Flask----------------------------------------
from flask import Flask, Response
from flask import render_template

app = Flask(__name__)

@app.route('/')
def index():
    #return "Default Message"
    return render_template("index.html")

@app.route('/video_feed')
def video_feed():
    #global cap
    return Response(main(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
                    
#------------------------------------------
def Distance():
    GPIO.output(TrigPin,GPIO.LOW)
    time.sleep(0.000002)
    GPIO.output(TrigPin,GPIO.HIGH)
    time.sleep(0.000015)
    GPIO.output(TrigPin,GPIO.LOW)

    t3 = time.time()

    while not GPIO.input(EchoPin):
        t4 = time.time()
        if (t4 - t3) > 0.03 :
            return -1
    t1 = time.time()
    while GPIO.input(EchoPin):
        t5 = time.time()
        if(t5 - t1) > 0.03 :
            return -1

    t2 = time.time()
    #time.sleep(0.01)
    #print ("distance_1 is %d " % (((t2 - t1)* 340 / 2) * 100))
    return ((t2 - t1)* 340 / 2) * 100
def Distance_test():
    num = 0
    ultrasonic = []
    while num < 5:
            distance = Distance()
            #print("distance is %f"%(distance) )
            while int(distance) == -1 :
                distance = Distance()
                #print("Tdistance is %f"%(distance) )
            while (int(distance) >= 500 or int(distance) == 0) :
                distance = Distance()
                #print("Edistance is %f"%(distance) )
            ultrasonic.append(distance)
            num = num + 1
            #time.sleep(0.01)
    #print ('ultrasonic')
    distance = (ultrasonic[1] + ultrasonic[2] + ultrasonic[3])/3
    #print("distance is %f"%(distance) ) 
    return distance

def avoid():
    distance = Distance_test()
    if distance < 20 :
        car.Car_Stop() 
        time.sleep(0.1)
        car.Car_Spin_Right(100,100) 
        time.sleep(0.5)
    else:
        car.Car_Run(65,69) 
        
        
def track_object(objs,labels):
   
    global x_deviation, y_max, tolerance, arr_track_data
    
    if(len(objs)==0):
        print("no objects to track")
        car.Car_Stop()
        arr_track_data=[0,0,0,0,0,0]
        return
    
    flag=0
    for obj in objs:
        lbl=labels.get(obj.id, obj.id)
        if (lbl==object_to_track):
            x_min, y_min, x_max, y_max = list(obj.bbox)
            flag=1
            break
        
    #print(x_min, y_min, x_max, y_max)
    if(flag==0):
        print("selected object no present")
        return
        
    x_diff=x_max-x_min
    y_diff=y_max-y_min
    print("x_diff: ",round(x_diff,5))
    print("y_diff: ",round(y_diff,5))
        
        
    obj_x_center=x_min+(x_diff/2)
    obj_x_center=round(obj_x_center,3)
    
    obj_y_center=y_min+(y_diff/2)
    obj_y_center=round(obj_y_center,3)
    
        
    x_deviation=round(0.5-obj_x_center,3)
    y_max=round(y_max,3)
        
    print("{",x_deviation,y_max,"}")
   
    thread = Thread(target = move_robot)
    thread.start()
    
    arr_track_data[0]=obj_x_center
    arr_track_data[1]=obj_y_center
    arr_track_data[2]=x_deviation
    arr_track_data[3]=y_max
    

def move_robot():
    global x_deviation, y_max, tolerance, arr_track_data,FallState
    
    y=1-y_max #distance from bottom of the frame
    if FallState == "Falled":
        cmd="FallDetection"
        car.Car_Stop()
    else:
        if(abs(x_deviation)<tolerance):
            delay1=0
            if(y<0.13):
                cmd="Stop"
                car.Car_Stop()
            else:
                cmd="forward"
                avoid()
                car.Car_Run(45,49) 

        else:
            if(x_deviation>=tolerance):
                cmd="Move Left"
                delay1=get_delay(x_deviation)

                car.Car_Left(30,160)
                time.sleep(delay1)
                car.Car_Stop()

            if(x_deviation<=-1*tolerance):
                cmd="Move Right"
                delay1=get_delay(x_deviation)

                car.Car_Right(160,30)
                time.sleep(delay1)
                car.Car_Stop()

    arr_track_data[4]=cmd
    arr_track_data[5]=delay1

def get_delay(deviation):
    deviation=abs(deviation)
    if(deviation>=0.4):
        d=0.6
    elif(deviation>=0.35 and deviation<0.40):
        d=0.050
    elif(deviation>=0.20 and deviation<0.35):
        d=0.040
    else:
        d=0.030
    return d

def warnFunction():
    #Motivate the buzzers
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(32, GPIO.OUT)
    p = GPIO.PWM(32, 440) 
    p.start(50)
    email.send_email()
    try:
        while 1:
            for dc in range(0, 101, 5):
                print ('start_1')
                p.ChangeDutyCycle(dc)
                time.sleep(1.0)
            for dc in range(100, -1, -5):
                p.ChangeDutyCycle(dc)
                print ('start_2')
                time.sleep(1.0)
    except KeyboardInterrupt:
        pass

def FallDetection():
    global FallState
    global scale,count

    # Background remove
    fg = cv2.createBackgroundSubtractorMOG2()

    while True:
        time.sleep(0.09)
        ret, img = cap.read()
        if not ret: break
        image = img.copy()

        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        gray = cv2.cvtColor(blurred, cv2.COLOR_RGB2GRAY)
        xgrad = cv2.Sobel(gray, cv2.CV_16SC1, 1, 0) 
        ygrad = cv2.Sobel(gray, cv2.CV_16SC1, 0, 1) 
        edge_output = cv2.Canny(xgrad, ygrad, 50, 150)

        fgmask = fg.apply(edge_output)

        hline = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 4), (-1, -1))
        vline = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1), (-1, -1))
        result = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, hline)
        result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, vline)

        dilateim = cv2.dilate(result, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)), iterations=1)

        contours, hier = cv2.findContours(dilateim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for c in contours:
            if cv2.contourArea(c) > 1000:
                (x, y, w, h) = cv2.boundingRect(c)
                if scale == 0: scale = -1;break
                scale = w / h
                cv2.putText(image, "scale:{:.3f}".format(scale), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.drawContours(image, [c], -1, (255, 0, 0), 1)
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 1)
                image = cv2.fillPoly(image, [c], (255, 255, 255)) 

        if scale > 0 and scale < 1:
            FallState = "Walking"
            cv2.putText(img, "Walking ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if scale > 0.9 and scale < 1.5:
            FallState = "Falling"
            cv2.putText(img, "Falling ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if scale > 1.6:
            if count==0:
                count =count+1
            if count>2:
                FallState = "Falled"
                cv2.putText(img, "Falled", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                count = 0

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()

def main():
    global FallState
    edgetpu = 0
    mdl = model
    
    interpreter, labels =cm.load_model(model_dir,mdl,lbl,edgetpu)
    
    fps=1
    arr_dur=[0,0,0]
    
    while True:
        start_time=time.time()
        
        #----------------Capture Camera Frame-----------------
        start_t0=time.time()
        ret, frame = cap.read()
        if not ret:
            break
        
        cv2_im = frame

        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        pil_im = Image.fromarray(cv2_im_rgb)
       
        arr_dur[0]=time.time() - start_t0
       
        #-------------------Inference---------------------------------
        start_t1=time.time()
        cm.set_input(interpreter, pil_im)
        interpreter.invoke()
        objs = cm.get_output(interpreter, score_threshold=threshold, top_k=top_k)
        
        arr_dur[1]=time.time() - start_t1
        #----------------------------------------------------
        if FallState == "Falled":
            warnFunction()
       #-----------------other------------------------------------
        start_t2=time.time()
        track_object(objs,labels)#tracking  <<<<<<<
       
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        cv2_im = append_text_img1(cv2_im, objs, labels, arr_dur, arr_track_data, FallState)
        
        ret, jpeg = cv2.imencode('.jpg', cv2_im)
        pic = jpeg.tobytes()
        
        #Flask streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + pic + b'\r\n\r\n')
       
        arr_dur[2]=time.time() - start_t2
        fps = round(1.0 / (time.time() - start_time),1)
        print("*********FPS: ",fps,"************")

    cap.release()
    cv2.destroyAllWindows()

def append_text_img1(cv2_im, objs, labels, arr_dur, arr_track_data,FallState):
    height, width, channels = cv2_im.shape
    font=cv2.FONT_HERSHEY_SIMPLEX
    
    global tolerance,scale
    
    #draw black rectangle on top
    cv2_im = cv2.rectangle(cv2_im, (0,0), (width, 24), (0,0,0), -1)
   
    #write processing durations
    cam=round(arr_dur[0]*1000,0)
    inference=round(arr_dur[1]*1000,0)
    other=round(arr_dur[2]*1000,0)
    
    #Write movement state
#     cv2.im = cv2.putText(cv2_im, FallState, (int(width/2), 20),font, 0.7, (150, 150, 255), 2)
#     cv2.putText(cv2_im, "scale:{:.3f}".format(scale), (int(width/2)-200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    #write FPS 
    total_duration=cam+inference+other
    fps=round(1000/total_duration,1)
    text1 = 'FPS: {}'.format(fps)
    cv2_im = cv2.putText(cv2_im, text1, (10, 20),font, 0.7, (150, 150, 255), 2)
   
    
    #draw black rectangle at bottom
    cv2_im = cv2.rectangle(cv2_im, (0,height-24), (width, height), (0,0,0), -1)
    
    #write deviations and tolerance
    str_tol='Tol : {}'.format(tolerance)
#     cv2_im = cv2.putText(cv2_im, str_tol, (10, height-8),font, 0.55, (150, 150, 255), 2)
  
    x_dev=arr_track_data[2]
    str_x='X: {}'.format(x_dev)
    if(abs(x_dev)<tolerance):
        color_x=(0,255,0)
    else:
        color_x=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_x, (10, height-8),font, 0.55, color_x, 2)
    
    y_dev=arr_track_data[3]
    str_y='Y: {}'.format(y_dev)
    if(abs(y_dev)>0.9):
        color_y=(0,255,0)
    else:
        color_y=(0,0,255)
    cv2_im = cv2.putText(cv2_im, str_y, (110, height-8),font, 0.55, color_y, 2)
   
    #write command, tracking status and speed
    cmd=arr_track_data[4]
    cv2_im = cv2.putText(cv2_im, str(cmd), (int(width/2) + 10, height-8),font, 0.68, (0, 255, 255), 2)
    
    delay1=arr_track_data[5]
    str_sp='Speed: {}%'.format(round(delay1/(0.1)*100,1))
#     cv2_im = cv2.putText(cv2_im, str_sp, (int(width/2) + 185, height-8),font, 0.55, (150, 150, 255), 2)
    
    if(cmd==0):
        str1="No object"
    elif(cmd=='Stop'):
        str1='Acquired'
    else:
        str1='Tracking'
    cv2_im = cv2.putText(cv2_im, str1, (width-140, 18),font, 0.7, (0, 255, 255), 2)
    
    #draw center cross lines
    cv2_im = cv2.rectangle(cv2_im, (0,int(height/2)-1), (width, int(height/2)+1), (255,0,0), -1)
    cv2_im = cv2.rectangle(cv2_im, (int(width/2)-1,0), (int(width/2)+1,height), (255,0,0), -1)
    
    #draw the center red dot on the object
    cv2_im = cv2.circle(cv2_im, (int(arr_track_data[0]*width),int(arr_track_data[1]*height)), 7, (0,0,255), -1)

    #draw the tolerance box
#     cv2_im = cv2.rectangle(cv2_im, (int(width/2-tolerance*width),0), (int(width/2+tolerance*width),height), (0,255,0), 2)
    
    for obj in objs:
        x0, y0, x1, y1 = list(obj.bbox)
        x0, y0, x1, y1 = int(x0*width), int(y0*height), int(x1*width), int(y1*height)
        percent = int(100 * obj.score)
        
        box_color, text_color, thickness=(0,150,255), (0,255,0),1
        

        text3 = '{}% {}'.format(percent, labels.get(obj.id, obj.id))
        
        if(labels.get(obj.id, obj.id)=="person"):
            cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), box_color, thickness)
            cv2_im = cv2.putText(cv2_im, text3, (x0, y1-5),font, 0.5, text_color, thickness)
            
    return cv2_im

if __name__ == '__main__':
    threadFall = threading.Thread(target=FallDetection)
    threadFall.start()
    app.run(host='192.168.0.115', port=5000, threaded=True) # Run FLASK
    main()