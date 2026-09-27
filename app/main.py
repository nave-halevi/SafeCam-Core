import cv2
import numpy as np  # ספרייה למתמטיקה ומטריצות

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Camera is running. Press 'q' to quit.")

    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)

    # יצירת "מברשת" לניפוח (אליפסה בגודל 7 על 7 פיקסלים)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        # שלב 1: החלת מחסר הרקע
        fg_mask = back_sub.apply(frame)

        # --- השלב החדש: ניקוי וחיבור המסכה ---
        # א. העלמת הצללים: כל מה שלא לבן בוהק (מעל 254) הופך לשחור (0)
        _, fg_mask = cv2.threshold(fg_mask, 254, 255, cv2.THRESH_BINARY)

        # ב. ניפוח הכתמים הלבנים (Dilation) כדי לחבר אזורים קרובים. 
        # iterations=4 אומר שאנחנו עושים את פעולת הניפוח 4 פעמים ברצף.
        fg_mask = cv2.dilate(fg_mask, kernel, iterations=4)
        # -------------------------------------

        # שלב 2: מציאת קווי המתאר
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # הגדלנו קצת את השטח המינימלי, כי עכשיו הכתמים התנפחו
            area = cv2.contourArea(contour)
            if area < 4000: 
                continue

            # שלב 3: ציור מלבן חוסם
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('SafeCam - Object Tracking', frame)
        cv2.imshow('SafeCam - Foreground Mask', fg_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()