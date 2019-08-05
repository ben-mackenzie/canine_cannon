try:
    import cv2
except Exception as e:
    print("Warning: OpenCV not installed. To use motion detection, make sure you've properly configured OpenCV.")

import time
import thread
import threading
import atexit
import sys
import termios
import contextlib

import imutils
import RPi.GPIO as GPIO
from Adafruit_MotorHAT import Adafruit_MotorHAT, Adafruit_DCMotor, Adafruit_StepperMotor


### User Parameters ###

MOTOR_X_REVERSED = False
MOTOR_Y_REVERSED = False

MAX_STEPS_X = 30
MAX_STEPS_Y = 15

RELAY_PIN = 22

#######################


@contextlib.contextmanager
def raw_mode(file):
    """
    Magic function that allows key presses.
    :param file:
    :return:
    """
    old_attrs = termios.tcgetattr(file.fileno())
    new_attrs = old_attrs[:]
    new_attrs[3] = new_attrs[3] & ~(termios.ECHO | termios.ICANON)
    try:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, new_attrs)
        yield
    finally:
        termios.tcsetattr(file.fileno(), termios.TCSADRAIN, old_attrs)


class VideoUtils(object):
    """
    Helper functions for video utilities.
    """
    @staticmethod
    def live_video(camera_port=0):
        """
        Opens a window with live video.
        :param camera:
        :return:
        """

        video_capture = cv2.VideoCapture(camera_port)

        while True:
            # Capture frame-by-frame
            ret, frame = video_capture.read()

            # Display the resulting frame
            cv2.imshow('Video', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything is done, release the capture
        video_capture.release()
        cv2.destroyAllWindows()

    @staticmethod
    def find_motion(callback, camera_port=0, show_video=False):

        camera = cv2.VideoCapture(camera_port)
        time.sleep(0.25)

        # initialize the first frame in the video stream
        firstFrame = None
        tempFrame = None
        count = 0

        # loop over the frames of the video
        while True:
            # grab the current frame and initialize the occupied/unoccupied
            # text

            (grabbed, frame) = camera.read()

            # if the frame could not be grabbed, then we have reached the end
            # of the video
            if not grabbed:
                break

            # resize the frame, convert it to grayscale, and blur it
            frame = imutils.resize(frame, width=500)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)

            # if the first frame is None, initialize it
            if firstFrame is None:
                print "Waiting for video to adjust..."
                if tempFrame is None:
                    tempFrame = gray
                    continue
                else:
                    delta = cv2.absdiff(tempFrame, gray)
                    tempFrame = gray
                    tst = cv2.threshold(delta, 5, 255, cv2.THRESH_BINARY)[1]
                    tst = cv2.dilate(tst, None, iterations=2)
                    if count > 30:
                        print "Done.\n Waiting for motion."
                        if not cv2.countNonZero(tst) > 0:
                            firstFrame = gray
                        else:
                            continue
                    else:
                        count += 1
                        continue

            # compute the absolute difference between the current frame and
            # first frame
            frameDelta = cv2.absdiff(firstFrame, gray)
            thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]

            # dilate the thresholded image to fill in holes, then find contours
            # on thresholded image
            thresh = cv2.dilate(thresh, None, iterations=2)
            c = VideoUtils.get_best_contour(thresh.copy(), 5000)

            if c is not None:
                
                ###### classify image and test for dog ######
                
                classes = None

                with open(args.classes, 'r') as f:
                    classes = [line.strip() for line in f.readlines()]

                COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

                net = cv2.dnn.readNet(args.weights, args.config)

                blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

                net.setInput(blob)

                outs = net.forward(get_output_layers(net))

                class_ids = []
                confidences = []
                boxes = []
                conf_threshold = 0.5
                nms_threshold = 0.4


                for out in outs:
                    for detection in out:
                        scores = detection[5:]
                        class_id = np.argmax(scores)
                        confidence = scores[class_id]
                        if confidence > 0.5:
                            center_x = int(detection[0] * Width)
                            center_y = int(detection[1] * Height)
                            w = int(detection[2] * Width)
                            h = int(detection[3] * Height)
                            x = center_x - w / 2
                            y = center_y - h / 2
                            class_ids.append(class_id)
                            confidences.append(float(confidence))
                            boxes.append([x, y, w, h])


                indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

                for i in indices:
                    i = i[0]
                    box = boxes[i]
                    x = box[0]
                    y = box[1]
                    w = box[2]
                    h = box[3]
                    draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))
                
                #### check for tigger using dog or cat classes
                dog_index = None
                detected_classes = [classes[class_id] for class_id in class_ids]
                if "dog" in detected_classes:
                    dog_index = detected_classes.index("dog")
                elif "cat" in detected_classes:
                    dog_index = detected_classes.index("dog")
                if dog_index != None and "couch" in detected_classes:
                    couch_index = detected_classes.index("couch")

                    #### compare center_y for dog and couch
                    dog_center = boxes[dog_index][1] + boxes[dog_index][3]/2
                    couch_center = boxes[couch_index][1] + boxes[couch_index][3] / 2
                    if dog_center > couch_center:

                        #### send center coordinates to turret
                    
                    
                
                
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                callback(c, frame)

            # show the frame and record if the user presses a key
            if show_video:
                cv2.imshow("Security Feed", frame)
                key = cv2.waitKey(1) & 0xFF

                # if the `q` key is pressed, break from the lop
                if key == ord("q"):
                    break

        # cleanup the camera and close any open windows
        camera.release()
        cv2.destroyAllWindows()

    @staticmethod
    def get_best_contour(imgmask, threshold):
        im, contours, hierarchy = cv2.findContours(imgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best_area = threshold
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > best_area:
                best_area = area
                best_cnt = cnt
        return best_cnt


class Turret(object):
    """
    Class used for turret control.
    """
    def __init__(self, friendly_mode=True):
        self.friendly_mode = friendly_mode

        # create a default object, no changes to I2C address or frequency
        self.mh = Adafruit_MotorHAT()
        atexit.register(self.__turn_off_motors)

        # Stepper motor 1
        self.sm_x = self.mh.getStepper(200, 1)      # 200 steps/rev, motor port #1
        self.sm_x.setSpeed(5)                       # 5 RPM
        self.current_x_steps = 0

        # Stepper motor 2
        self.sm_y = self.mh.getStepper(200, 2)      # 200 steps/rev, motor port #2
        self.sm_y.setSpeed(5)                       # 5 RPM
        self.current_y_steps = 0

        # Relay
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(RELAY_PIN, GPIO.OUT)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    def calibrate(self):
        """
        Waits for input to calibrate the turret's axis
        :return:
        """
        print "Please calibrate the tilt of the gun so that it is level. Commands: (w) moves up, " \
              "(s) moves down. Press (enter) to finish.\n"
        self.__calibrate_y_axis()

        print "Please calibrate the yaw of the gun so that it aligns with the camera. Commands: (a) moves left, " \
              "(d) moves right. Press (enter) to finish.\n"
        self.__calibrate_x_axis()

        print "Calibration finished."

    def __calibrate_x_axis(self):
        """
        Waits for input to calibrate the x axis
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Turret.move_backward(self.sm_x, 5)
                        else:
                            Turret.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Turret.move_forward(self.sm_x, 5)
                        else:
                            Turret.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print "Error: Unable to calibrate turret. Exiting..."
                sys.exit(1)

    def __calibrate_y_axis(self):
        """
        Waits for input to calibrate the y axis.
        :return:
        """
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch:
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Turret.move_forward(self.sm_y, 5)
                        else:
                            Turret.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Turret.move_backward(self.sm_y, 5)
                        else:
                            Turret.move_forward(self.sm_y, 5)
                    elif ch == "\n":
                        break

            except (KeyboardInterrupt, EOFError):
                print "Error: Unable to calibrate turret. Exiting..."
                sys.exit(1)

    def motion_detection(self, show_video=False):
        """
        Uses the camera to move the turret. OpenCV ust be configured to use this.
        :return:
        """
        VideoUtils.find_motion(self.__move_axis, show_video=show_video)

    def __move_axis(self, contour, frame):
        (v_h, v_w) = frame.shape[:2]
        (x, y, w, h) = cv2.boundingRect(contour)

        # find height
        target_steps_x = (2*MAX_STEPS_X * (x + w / 2) / v_w) - MAX_STEPS_X
        target_steps_y = (2*MAX_STEPS_Y*(y+h/2) / v_h) - MAX_STEPS_Y

        print "x: %s, y: %s" % (str(target_steps_x), str(target_steps_y))
        print "current x: %s, current y: %s" % (str(self.current_x_steps), str(self.current_y_steps))

        t_x = threading.Thread()
        t_y = threading.Thread()
        t_fire = threading.Thread()

        # move x
        if (target_steps_x - self.current_x_steps) > 0:
            self.current_x_steps += 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Turret.move_forward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Turret.move_backward, args=(self.sm_x, 2,))
        elif (target_steps_x - self.current_x_steps) < 0:
            self.current_x_steps -= 1
            if MOTOR_X_REVERSED:
                t_x = threading.Thread(target=Turret.move_backward, args=(self.sm_x, 2,))
            else:
                t_x = threading.Thread(target=Turret.move_forward, args=(self.sm_x, 2,))

        # move y
        if (target_steps_y - self.current_y_steps) > 0:
            self.current_y_steps += 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Turret.move_backward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Turret.move_forward, args=(self.sm_y, 2,))
        elif (target_steps_y - self.current_y_steps) < 0:
            self.current_y_steps -= 1
            if MOTOR_Y_REVERSED:
                t_y = threading.Thread(target=Turret.move_forward, args=(self.sm_y, 2,))
            else:
                t_y = threading.Thread(target=Turret.move_backward, args=(self.sm_y, 2,))

        # fire if necessary
        if not self.friendly_mode:
            if abs(target_steps_y - self.current_y_steps) <= 2 and abs(target_steps_x - self.current_x_steps) <= 2:
                t_fire = threading.Thread(target=Turret.fire)

        t_x.start()
        t_y.start()
        t_fire.start()

        t_x.join()
        t_y.join()
        t_fire.join()

    def interactive(self):
        """
        Starts an interactive session. Key presses determine movement.
        :return:
        """

        Turret.move_forward(self.sm_x, 1)
        Turret.move_forward(self.sm_y, 1)

        print 'Commands: Pivot with (a) and (d). Tilt with (w) and (s). Exit with (q)\n'
        with raw_mode(sys.stdin):
            try:
                while True:
                    ch = sys.stdin.read(1)
                    if not ch or ch == "q":
                        break

                    if ch == "w":
                        if MOTOR_Y_REVERSED:
                            Turret.move_forward(self.sm_y, 5)
                        else:
                            Turret.move_backward(self.sm_y, 5)
                    elif ch == "s":
                        if MOTOR_Y_REVERSED:
                            Turret.move_backward(self.sm_y, 5)
                        else:
                            Turret.move_forward(self.sm_y, 5)
                    elif ch == "a":
                        if MOTOR_X_REVERSED:
                            Turret.move_backward(self.sm_x, 5)
                        else:
                            Turret.move_forward(self.sm_x, 5)
                    elif ch == "d":
                        if MOTOR_X_REVERSED:
                            Turret.move_forward(self.sm_x, 5)
                        else:
                            Turret.move_backward(self.sm_x, 5)
                    elif ch == "\n":
                        Turret.fire()

            except (KeyboardInterrupt, EOFError):
                pass

    @staticmethod
    def fire():
        GPIO.output(RELAY_PIN, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(RELAY_PIN, GPIO.LOW)

    @staticmethod
    def move_forward(motor, steps):
        """
        Moves the stepper motor forward the specified number of steps.
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.FORWARD,  Adafruit_MotorHAT.INTERLEAVE)

    @staticmethod
    def move_backward(motor, steps):
        """
        Moves the stepper motor backward the specified number of steps
        :param motor:
        :param steps:
        :return:
        """
        motor.step(steps, Adafruit_MotorHAT.BACKWARD, Adafruit_MotorHAT.INTERLEAVE)

    def __turn_off_motors(self):
        """
        Recommended for auto-disabling motors on shutdown!
        :return:
        """
        self.mh.getMotor(1).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(2).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(3).run(Adafruit_MotorHAT.RELEASE)
        self.mh.getMotor(4).run(Adafruit_MotorHAT.RELEASE)

if __name__ == "__main__":
    t = Turret(friendly_mode=False)

    user_input = raw_input("Choose an input mode: (1) Motion Detection, (2) Interactive\n")

    if user_input == "1":
        t.calibrate()
        if raw_input("Live video? (y, n)\n").lower() == "y":
            t.motion_detection(show_video=True)
        else:
            t.motion_detection()
    elif user_input == "2":
        if raw_input("Live video? (y, n)\n").lower() == "y":
            thread.start_new_thread(VideoUtils.live_video, ())
        t.interactive()
    else:
        print "Unknown input mode. Please choose a number (1) or (2)"
