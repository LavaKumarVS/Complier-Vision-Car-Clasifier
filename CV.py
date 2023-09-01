import cv2
import time
import numpy as np
import csv

car_classifier = cv2.CascadeClassifier(r'C:\Users\Time Traveller\Desktop\Pratice DSA\Smart Parking System\OpenCV-Implementaion_master_haarcascade_car.xml')
cap = cv2.VideoCapture(r'C:\Users\Time Traveller\Desktop\Pratice DSA\Smart Parking System\SaveTube.io-Cars Moving On Road Stock Footage - Free Download-(1080p).mp4')
csv_file = open('car_detection.csv', 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(['index', 'Time', 'Num_Cars'])
frame_count = 0
while cap.isOpened():
    start_time = time.time()  
    time.sleep(.05)
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cars = car_classifier.detectMultiScale(gray, 1.4, 2)
    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
    num_cars = len(cars)
    cv2.putText(frame, f"Number of Cars: {num_cars}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Cars', frame)
    end_time = time.time()
    processing_time = end_time - start_time
    cv2.putText(frame, f"Processing Time: {processing_time:.2f} seconds", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    csv_writer.writerow([frame_count, processing_time, num_cars])
    frame_count += 1
    if cv2.waitKey(1) == 13: 
        break
csv_file.close()
cap.release()
cv2.destroyAllWindows()