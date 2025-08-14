import cv2
import mediapipe as mp
import time


# for this to run on Visual Studios Code, follow these steps
# 1. pip install opencv-python
# 2. pip install opencv-python mediapipe
# 3. python HandTracking.py

class handDetector():
    def __init__(self, mode=False,maxHands=2,detectionCon=0.5,trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon



        self.mphand = mp.solutions.hands
        self.hand = self.mphand.Hands(static_image_mode=self.mode, max_num_hands=self.maxHands, min_detection_confidence=self.detectionCon, min_tracking_confidence=self.trackCon)
        self.mp = mp.solutions.drawing_utils


    def findHands(self, img, draw=True):
        rgbhand = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.result = self.hand.process(rgbhand)

        if self.result.multi_hand_landmarks:
            for hrhand in self.result.multi_hand_landmarks:
                if draw:
                    self.mp.draw_landmarks(img, hrhand, self.mphand.HAND_CONNECTIONS)
        return img


    def findPosition(self, img, habdNo=0, draw=True):

        lmList = []
        if self.result.multi_hand_landmarks:
            myHand = self.result.multi_hand_landmarks[habdNo]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 15, (255, 0, 255), cv2.FILLED)

        return lmList




def main():
    ptime = 0
    ctime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            print(lmList[4])


        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (250, 0, 250), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()