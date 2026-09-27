import cv2
import mediapipe as mp

def main():
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("AI Camera is running. Press 'q' to quit.")

    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_rgb.flags.writeable = False
            results = holistic.process(image_rgb)
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # ציור השלד, הידיים והפנים (השארנו כדי להמשיך לראות שזה עובד)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, landmark_drawing_spec=None, connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(image_bgr, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())
            
            # --- הלוגיקה החדשה: זיהוי נפילה ---
            if results.pose_landmarks:
                # קודם כל מציירים את השלד עצמו
                mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                # שולפים את רשימת נקודות הציון (Landmarks)
                landmarks = results.pose_landmarks.landmark

                # שליפת קואורדינטות ה-Y של האף והאגן
                # MediaPipe מחזיר ערך בין 0.0 (למעלה) ל-1.0 (למטה)
                nose_y = landmarks[mp_holistic.PoseLandmark.NOSE.value].y
                left_hip_y = landmarks[mp_holistic.PoseLandmark.LEFT_HIP.value].y
                right_hip_y = landmarks[mp_holistic.PoseLandmark.RIGHT_HIP.value].y

                # מחשבים את ממוצע גובה האגן (כי לפעמים צד אחד מוסתר או בזווית)
                avg_hip_y = (left_hip_y + right_hip_y) / 2.0

                # בדיקת נפילה: אם האף (Y גדול) ירד מתחת לגובה האגן
                # אנחנו לוקחים מרווח ביטחון קטן (+0.1) כדי למנוע אזעקות שווא כשאדם סתם מתכופף
                if nose_y > (avg_hip_y - 0.1):
                    # מדפיסים התראת חירום על המסך באדום
                    cv2.putText(image_bgr, "FALL DETECTED!", (50, 100), 
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

            cv2.imshow('SafeCam - AI Tracking (Pose, Hands, Face)', image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()