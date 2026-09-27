import cv2

def main():
    # 0 מציין את מצלמת הרשת הדיפולטית של המחשב (אם יש לך כמה, אפשר לנסות 1 או 2)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera is running. Press 'q' to quit.")

    # הלולאה הזו רצה ברצף - זה ה-FPS שלנו שמייצר אשליית וידאו!
    while True:
        # קריאת פריים בודד (מטריצת המספרים של התמונה הנוכחית)
        ret, frame = cap.read()

        # אם אי אפשר לקרוא את הפריים (המצלמה התנתקה למשל), עוצרים
        if not ret:
            print("Error: Could not read frame.")
            break

        # מציג את הפריים (המטריצה) על המסך בחלון עם כותרת
        cv2.imshow('SafeCam - Frame by Frame', frame)

        # מחכה מילישנייה אחת, ואם המשתמש לחץ על המקש 'q' במקלדת - עוצרים את הלולאה
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # בסיום הלולאה: שחרור המצלמה וסגירת כל החלונות בצורה נקייה
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()