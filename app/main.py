import cv2
import mediapipe as mp

def main():
    # אתחול הכלים של MediaPipe
    mp_holistic = mp.solutions.holistic
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("AI Camera is running. Press 'q' to quit.")

    # הפעלת המודל עם הגדרות רגישות (0.5 = 50% ביטחון כדי לזהות ולעקוב)
    with mp_holistic.Holistic(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as holistic:
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            # המצלמה של ה-OpenCV קוראת צבעים בפורמט BGR, אבל MediaPipe דורש RGB
            # לכן אנחנו ממירים את הצבעים לפני ששולחים ל-AI
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # נעילת התמונה לכתיבה כדי לשפר ביצועים בזמן שה-AI מנתח אותה
            image_rgb.flags.writeable = False
            
            # --- כאן קורה הקסם: ה-AI מנתח את התמונה ומוצא את כל הנקודות ---
            results = holistic.process(image_rgb)
            
            # החזרת התמונה למצב כתיבה והמרה חזרה ל-BGR כדי שנוכל להציג אותה
            image_rgb.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # 1. ציור רשת הפנים (Face Mesh)
            if results.face_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.face_landmarks,
                    mp_holistic.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

            # 2. ציור כף יד ימין
            if results.right_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.right_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

            # 3. ציור כף יד שמאל
            if results.left_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.left_hand_landmarks,
                    mp_holistic.HAND_CONNECTIONS,
                    connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style())

            # 4. ציור שלד הגוף (Pose)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image_bgr,
                    results.pose_landmarks,
                    mp_holistic.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            # תצוגת התוצאה
            cv2.imshow('SafeCam - AI Tracking (Pose, Hands, Face)', image_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()