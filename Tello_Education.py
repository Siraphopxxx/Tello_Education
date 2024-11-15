
from tkinter import Tk, Label, Button, Frame, Scale
import cv2
from PIL import Image, ImageTk  # make imports from the Pillow library for displaying the video stream with Tkinter.
from djitellopy import Tello
import threading # Import threading for our takeoff/land method
import time
import mediapipe as mp
import time
import numpy as np



class Drone:
        
        class Control:
                def __init__(self):

                        self.root = Tk()
                        self.root.title("Drone Keyboard Controller - Tkinter")
                        self.root.minsize(800, 600)

                        self.input_frame = Frame(self.root)

                        self.drone = Tello()
                        self.drone.connect()
                        self.drone.streamon()
                        self.frame = self.drone.get_frame_read()

                        self.drone.speed = 80

                        self.speed_bar = Scale(self.root, from_=25, to=100, length=150, tickinterval=25,
                                                digits=3, label='Drone Flight Speed:',
                                                resolution=25, showvalue=False, orient="horizontal")
                        self.speed_bar.set(50)
                        self.set_speed_btn = Button(self.root, text='Reset Speed', command=self.updateSpeed)
                        self.cap_lbl = Label(self.root)
                        self.takeoff_land_button = Button(self.root, text="Takeoff/Land", command=lambda: self.takeoff_land())
                        self.run_app()

                def updateSpeed(self):
                        self.drone.speed = self.speed_bar.get()
                        print(f'reset speed to {self.drone.speed:.1f}')
                        

                def takeoff_land(self):
                        if self.drone.is_flying:
                                threading.Thread(target=lambda: self.drone.land()).start()
                        else:
                                threading.Thread(target=lambda: self.drone.takeoff()).start()

                def fly(self,direction, drone):

                        drone.send_rc_control(direction[0], direction[1], direction[2], direction[3])
                        time.sleep(0.05)

                def start_flying(self,event,direction, drone, speed):

                        lr, fb, ud, yv = 0, 0, 0, 0

                        if direction == "upward":
                                print("Moving up")
                                ud = speed
                        elif direction == "downward":
                                ud = -speed
                                print("Moving down")
                        elif direction == "forward":
                                fb = speed
                                print("Moving forward")
                        elif direction == "backward":
                                fb = -speed
                                print("Moving backward")
                        elif direction == "yaw_left":
                                yv = -speed
                                print("turning left")
                        elif direction == "yaw_right":
                                yv = speed
                                print("turning right")
                        elif direction == "left":
                                lr = -speed
                                print("Moving left")
                        elif direction == "right":
                                lr = speed
                                print("Moving right")

                        if [lr, fb, ud, yv] != [0, 0, 0, 0]:
                                threading.Thread(target=lambda: self.fly([lr, fb, ud, yv], drone)).start()

                def stop_flying(self,event, drone):
                        """When user releases a movement key the drone stops performing that movement"""
                        drone.send_rc_control(0, 0, 0, 0)
                                

                def run_app(self):
                        try:
                                self.takeoff_land_button.pack(side='bottom', pady=10)

                                self.input_frame.bind('<KeyPress-w>', lambda event: self.start_flying(event, 'upward', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-w>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-a>', lambda event: self.start_flying(event, 'yaw_left', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-a>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-s>', lambda event: self.start_flying(event, 'downward', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-s>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-d>', lambda event: self.start_flying(event, 'yaw_right', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-d>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-Up>', lambda event: self.start_flying(event, 'forward', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-Up>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-Down>', lambda event: self.start_flying(event, 'backward', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-Down>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-Left>', lambda event: self.start_flying(event, 'left', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-Left>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.bind('<KeyPress-Right>', lambda event: self.start_flying(event, 'right', self.drone, self.drone.speed))
                                self.input_frame.bind('<KeyRelease-Right>', lambda event: self.stop_flying(event, self.drone))

                                self.input_frame.pack()
                                self.input_frame.focus_set()

                                self.cap_lbl.pack(anchor="center")

                                self.speed_bar.pack(anchor='w', padx=(10, 0))
                                self.set_speed_btn.pack(anchor='sw', padx=(10, 0), pady=(20, 0))

                                self.video_stream()

                                self.root.mainloop()

                        except Exception as e:
                                print(f"Error running the application: {e}")
                        finally:
                                self.cleanup()

                def video_stream(self):
                        h = 480
                        w = 720       

                        frame = self.frame.frame

                        frame = cv2.resize(frame, (w, h))

                        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

                        img = Image.fromarray(cv2image)

                        imgtk = ImageTk.PhotoImage(image=img)

                        self.cap_lbl.pack(anchor="center", pady=15)

                        self.cap_lbl.imgtk = imgtk

                        self.cap_lbl.configure(image=imgtk)

                        self.cap_lbl.after(5, self.video_stream)

                def cleanup(self) -> None:
                        try:
                                print("Cleaning up resources...")
                                self.drone.end()
                                self.root.quit()  
                        except Exception as e:
                                print(f"Error performing cleanup: {e}")



        class Detect:
                def __init__(self):
                        self.width, self.height = 720, 480
                        self.xPID, self.yPID, self.zPID = [0.21, 0, 0.1], [0.27, 0, 0.1], [0.0021, 0, 0.1]
                        self.xTarget, self.yTarget, self.zTarget = self.width // 2, self.height // 2, 33000
                        self.pError, self.pTime, self.I = 0, 0, 0

                        self.mpPose = mp.solutions.pose
                        self.pose = self.mpPose.Pose()
                        self.mpHands = mp.solutions.hands
                        self.hands = self.mpHands.Hands()

                        self.mp_face_detection = mp.solutions.face_detection
                        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)

                        self.mpDraw = mp.solutions.drawing_utils

                        self.take_pic_allow = True

                        self.my_drone = Tello()
                        self.my_drone.connect()
                        print(self.my_drone.get_battery())
                        self.my_drone.streamoff()
                        self.my_drone.streamon()
                        self.my_drone.takeoff()
                        self.my_drone.move_up(60)
                        self.run()

                def PIDController(self, PID, img, target, cVal, limit=[-100, 100], pTime=0, pError=0, I=0, draw=False):
                        t = time.time() - pTime
                        error = target - cVal
                        P = PID[0] * error
                        I = I + (PID[1] * error * t)
                        D = PID[2] * (error - pError) / t

                        val = P + I + D
                        val = float(np.clip(val, limit[0], limit[1]))
                        if draw:
                            cv2.putText(img, str(int(val)), (50, 70), cv2.FONT_HERSHEY_PLAIN, 4, (255, 0, 255), 3)

                        pError = error
                        pTime = time.time()

                        return int(val)
                

                def run(self):
                        while True:
                                img = self.my_drone.get_frame_read().frame

                                img = cv2.resize(img, (self.width, self.height))

                                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                pose_results = self.pose.process(imgRGB)
                                hand_results = self.hands.process(imgRGB)
                                xVal, yVal, zVal,kVal = 0, 0, 0,0

                                if pose_results.pose_landmarks:
                                        self.mpDraw.draw_landmarks(img, pose_results.pose_landmarks, self.mpPose.POSE_CONNECTIONS)
                                        landmarks = pose_results.pose_landmarks.landmark

                                        left_shoulder = landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value] #เข้าถึง object ที่เก็บค่า xyz จากไหล่ ซ้าย
                                        right_shoulder = landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value] #เข้าถึง object ที่เก็บค่า xyz จากไหล่ ขวา
                                        left_hip = landmarks[self.mpPose.PoseLandmark.LEFT_HIP.value] #เข้าถึง object ที่เก็บค่า xyz จากสะโพก ซ้าย
                                        right_hip = landmarks[self.mpPose.PoseLandmark.RIGHT_HIP.value] #เข้าถึง object ที่เก็บค่า xyz จากสะโพก ขวา
                                        cx, cy = int((left_shoulder.x + right_shoulder.x) / 2 * self.width), int((left_shoulder.y + right_shoulder.y) / 2 * self.height) 
                                        area = abs((left_shoulder.x - right_shoulder.x)*self.width) * abs((left_hip.y + right_hip.y)*self.height / 2)

                                        left_elbow = landmarks[self.mpPose.PoseLandmark.LEFT_ELBOW.value] 
                                        right_elbow = landmarks[self.mpPose.PoseLandmark.RIGHT_ELBOW.value]
                                        left_wrist =  landmarks[self.mpPose.PoseLandmark.LEFT_WRIST.value]
                                        right_wrist = landmarks[self.mpPose.PoseLandmark.RIGHT_WRIST.value]
                                        left_shoulder = landmarks[self.mpPose.PoseLandmark.LEFT_SHOULDER.value] 
                                        right_shoulder = landmarks[self.mpPose.PoseLandmark.RIGHT_SHOULDER.value]

                                        # ตรวจสอบแขนซ้ายและขวา 
                                        left_arm = (left_wrist.x) < (left_shoulder.x) and (left_elbow.x) < (left_shoulder.x) and left_shoulder.x > right_shoulder.x
                                        right_arm = right_wrist.x > right_shoulder.x and right_elbow.x > right_shoulder.x and left_shoulder.x > right_shoulder.x

                                        if left_arm:
                                            cv2.putText(img, "LEFT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) 
                                            kVal = -50
                                        elif right_arm: 
                                            cv2.putText(img, "RIGHT", (50, 100), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2) 
                                            kVal  = 50


                                        xVal = self.PIDController(self.xPID, img, self.xTarget, cx)
                                        yVal = self.PIDController(self.yPID, img, self.yTarget, cy)
                                        zVal = self.PIDController(self.zPID, img, self.zTarget, area, limit=[-20, 15], draw=True)
                                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)


 
                                if hand_results.multi_hand_landmarks:
                                        for hand_landmarks in hand_results.multi_hand_landmarks:
                                            mp.solutions.drawing_utils.draw_landmarks(img, hand_landmarks, self.mpHands.HAND_CONNECTIONS)

                                            # Get landmarks
                                            landmarks = hand_landmarks.landmark
                                            index_finger_tip = landmarks[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                                            middle_finger_tip = landmarks[self.mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                                            ring_finger_tip = landmarks[self.mpHands.HandLandmark.RING_FINGER_TIP]
                                            pinky_tip = landmarks[self.mpHands.HandLandmark.PINKY_TIP]

                                            # Check if index and middle fingers are up and others are down
                                            if (index_finger_tip.y < landmarks[self.mpHands.HandLandmark.INDEX_FINGER_MCP].y and
                                                middle_finger_tip.y < landmarks[self.mpHands.HandLandmark.MIDDLE_FINGER_MCP].y and
                                                ring_finger_tip.y > landmarks[self.mpHands.HandLandmark.RING_FINGER_MCP].y and
                                                pinky_tip.y > landmarks[self.mpHands.HandLandmark.PINKY_MCP].y):

                                                if (img is not None and self.take_pic_allow):
                                                    cv2.imwrite(f'Resources/Images/{time.time()}.jpg',img)
                                                    cv2.putText(img, "CAPTURE", (250, 250), cv2.FONT_HERSHEY_PLAIN, 8, (0, 255, 0), 4) 
                                                    self.take_pic_allow = False 
                                                else: 
                                                    self.take_pic_allow = True
                                else : 
                                        self.take_pic_allow = True


                        
                                self.my_drone.send_rc_control(kVal,zVal, yVal, int(-xVal*2))
                                cv2.imshow('Image', img)


                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                        self.my_drone.land()
                                        break



        def __init__(self):
            self.drone = Tello()
            self.drone.connect()
            self.drone_onair = False

                # speed 10 - 100 cm/s
        def set_speed(self,speed):
                self.drone.setspeed(speed)

        def takeoff(self):
                self.drone.takeoff()
                self.drone_onair = True

        def land(self):
                self.drone.land()
                self.drone_onair = False

        # direction: up, down, left, right, forward or back distance: 20-500 cm
        def fly(self,direction:str,distance:int):
                if (self.drone_onair == True):
                    self.drone.move(direction,distance)

        #distance: 20-500 cm
        def fly_left(self,distance:int):
               if (self.drone_onair == True):
                    self.drone.move_left(distance)    
        def fly_right(self,distance:int):
                if (self.drone_onair == True):
                    self.drone.move_right(distance)

        #distance: 20-500 cm
        def fly_up(self,distance:int):
                if (self.drone_onair == True):
                    self.drone.move_up(distance)
        def fly_down(self,distance:int):
                if (self.drone_onair == True):
                    self.drone.move_down(distance)

        #distance: 20-500 cm
        def fly_forward(self,distance:int):
                if (self.drone_onair == True):
                    self.drone.move_forward(distance)
        def fly_back(self,distance:int):
                if (self.drone_onair == True):
                    self.drone.move_back(distance)


        # degree 1-360    
        def yaw_left(self,degree:int):
                if (self.drone_onair == True):
                    self.drone.rotate_counter_clockwise(degree)
        def yaw_right(self,degree:int):
                if (self.drone_onair == True):
                    self.drone.rotate_clockwise(degree)



        #direction: l (left), r (right), f (forward) or b (back)
        def flip(self,direction:str):
                if direction == "left":direction = "l"
                elif direction == "right":direction = "r"
                elif direction == "forward":direction = "f"
                elif direction == "back":direction = "b"

                if (self.drone_onair == True):
                    self.drone.flip(direction)

        def flip_left(self):
                if (self.drone_onair == True):
                    self.drone.flip_left()

        def flip_right(self):
                if (self.drone_onair == True):
                    self.drone.flip_right()

        def flip_forward(self):
                if (self.drone_onair == True):
                    self.drone.flip_forward()

        def flip_back(self):
                if (self.drone_onair == True):
                    self.drone.flip_back()


        #ex: pentagon have side:5 distance:20-500 cm
        def fly_geometric(self,side:int,distance:int):
                if(self.drone_onair == True):
                        interrior_angle = (side-2)*180/side
                        exterrior_angle = (180 - interrior_angle)
                        for i in range(side):
                                self.fly_forward(distance)
                                self.yaw_right(int(exterrior_angle))


        # x: -500-500|y: -500-500|z: -500-500|speed: 10-100
        def fly_xyz_speed(self, x: int, y: int, z: int, speed: int):
                if(self.drone_onair == True):
                    self.drone.go_xyz_speed(x,y,z,speed)


        """Fly to x2 y2 z2 in a curve via x2 y2 z2. Speed defines the traveling speed in cm/s.

        - Both points are relative to the current position
        - The current position and both points must form a circle arc.
        - If the arc radius is not within the range of 0.5-10 meters, it raises an Exception
        - x1/x2, y1/y2, z1/z2 can't both be between -20-20 at the same time, but can both be 0.

        Arguments:
                x1: -500-500
                x2: -500-500
                y1: -500-500
                y2: -500-500
                z1: -500-500
                z2: -500-500
                speed: 10-60
        """
        def fly_curve_xyz_speed(self,x1:int,y1:int,z1:int,x2:int,y2:int,z2:int,speed:int):    
                if(self.drone_onair == True):
                    self.drone.curve_xyz_speed(x1, y1, z1, x2, y2, z2, speed)
 