from datetime import datetime
import os
from deepface import DeepFace
import cv2
import pandas as pd

# Path to the folder containing known faces
path = 'Images'


def markAttendance(name):
  today_date = datetime.now().strftime('%Y-%m-%d')
  if not os.path.exists('Attendance'):
    os.makedirs('Attendance')

  filename = f'Attendance/Attendance_{today_date}.csv'
  now = datetime.now()
  time_string = now.strftime('%H:%M:%S')

  if os.path.exists(filename):
    df = pd.read_csv(filename)
    if name not in df['Name'].values:
      new_row = pd.DataFrame({'Name': [name], 'Time': [time_string]})
      df = pd.concat([df, new_row], ignore_index=True)
      df.to_csv(filename, index=False)
      print(f'Attendance Marked for: {name} at {time_string}')
  else:
    df = pd.DataFrame({'Name': [name], 'Time': [time_string]})
    df.to_csv(filename, index=False)
    print(f'Attendance Marked for: {name} at {time_string}')


print('Initializing Webcam...')
cap = cv2.VideoCapture(0)

counter = 0
recognized_name = 'Scanning...'

while True:
  success, img = cap.read()
  if not success:
    print('Failed to grab frame.')
    break

  counter += 1
  if counter % 30 == 0:
    try:
      dfs = DeepFace.find(
          img_path=img,
          db_path=path,
          enforce_detection=False,
          silent=True,
      )
      if len(dfs) > 0 and not dfs[0].empty:
        matched_path = dfs[0]['identity'].iloc[0]
        base_name = os.path.basename(matched_path)
        recognized_name = os.path.splitext(base_name)[0].upper()
        markAttendance(recognized_name)
      else:
        recognized_name = 'Unknown'
    except Exception as e:
      recognized_name = 'Scanning...'

  cv2.putText(
      img,
      f'Target: {recognized_name}',
      (50, 50),
      cv2.FONT_HERSHEY_SIMPLEX,
      1,
      (0, 255, 0),
      2,
  )

  # Use window flag to ensure compatibility on Windows
  cv2.namedWindow('Face Recognition Attendance System', cv2.WINDOW_NORMAL)
  cv2.imshow('Face Recognition Attendance System', img)

  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

cap.release()
cv2.destroyAllWindows()