import cv2
import mediapipe as mp
import numpy as np
import time

# ----------------------------------
# CAMERA
# ----------------------------------
cap = cv2.VideoCapture(0)

# Get camera size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# ----------------------------------
# MEDIAPIPE HAND DETECTION
# ----------------------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ----------------------------------
# DRAWING SETTINGS
# ----------------------------------
canvas = np.zeros((height, width, 3), dtype=np.uint8)

prev_x, prev_y = 0, 0

# Default colour - Green
draw_color = (0, 255, 0)

brush_size = 8
eraser_size = 50

eraser_mode = False
recording = False
video_writer = None

# Store previous canvas for undo
history = []


# ----------------------------------
# DRAW BUTTONS
# ----------------------------------
def draw_buttons(frame):

    # Red
    cv2.rectangle(frame, (10, 10), (90, 70), (0, 0, 255), -1)

    # Green
    cv2.rectangle(frame, (100, 10), (180, 70), (0, 255, 0), -1)

    # Blue
    cv2.rectangle(frame, (190, 10), (270, 70), (255, 0, 0), -1)

    # Yellow
    cv2.rectangle(frame, (280, 10), (360, 70), (0, 255, 255), -1)

    # Eraser
    cv2.rectangle(frame, (370, 10), (470, 70), (150, 150, 150), -1)
    cv2.putText(
        frame, "ERASER", (378, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45, (0, 0, 0), 1
    )

    # Clear
    cv2.rectangle(frame, (480, 10), (580, 70), (50, 50, 50), -1)
    cv2.putText(
        frame, "CLEAR", (495, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1
    )

    # Undo
    cv2.rectangle(frame, (590, 10), (690, 70), (100, 100, 100), -1)
    cv2.putText(
        frame, "UNDO", (605, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5, (255, 255, 255), 1
    )


# ----------------------------------
# MAIN PROGRAM
# ----------------------------------
while True:

    success, frame = cap.read()

    if not success:
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Resize canvas if needed
    if frame.shape != canvas.shape:
        height, width = frame.shape[:2]
        canvas = np.zeros_like(frame)

    # Draw buttons
    draw_buttons(frame)

    # Convert frame to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect hand
    results = hands.process(rgb)

    if results.multi_hand_landmarks:

        for hand_landmarks in results.multi_hand_landmarks:

            # Draw hand landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            # ----------------------------------
            # INDEX FINGER POSITION
            # ----------------------------------
            x = int(
                hand_landmarks.landmark[8].x * width
            )

            y = int(
                hand_landmarks.landmark[8].y * height
            )

            # Finger indicator
            indicator_color = (
                (0, 0, 0) if eraser_mode
                else draw_color
            )

            cv2.circle(
                frame,
                (x, y),
                10,
                indicator_color,
                -1
            )

            # ----------------------------------
            # BUTTON SELECTION
            # ----------------------------------

            # Red
            if 10 < x < 90 and 10 < y < 70:

                draw_color = (0, 0, 255)
                eraser_mode = False
                prev_x, prev_y = 0, 0

            # Green
            elif 100 < x < 180 and 10 < y < 70:

                draw_color = (0, 255, 0)
                eraser_mode = False
                prev_x, prev_y = 0, 0

            # Blue
            elif 190 < x < 270 and 10 < y < 70:

                draw_color = (255, 0, 0)
                eraser_mode = False
                prev_x, prev_y = 0, 0

            # Yellow
            elif 280 < x < 360 and 10 < y < 70:

                draw_color = (0, 255, 255)
                eraser_mode = False
                prev_x, prev_y = 0, 0

            # Eraser
            elif 370 < x < 470 and 10 < y < 70:

                eraser_mode = True
                prev_x, prev_y = 0, 0

            # Clear
            elif 480 < x < 580 and 10 < y < 70:

                history.append(canvas.copy())
                canvas = np.zeros_like(frame)
                prev_x, prev_y = 0, 0

            # Undo
            elif 590 < x < 690 and 10 < y < 70:

                if len(history) > 0:
                    canvas = history.pop()

                prev_x, prev_y = 0, 0

            # ----------------------------------
            # DRAW
            # ----------------------------------
            else:

                if prev_x == 0 and prev_y == 0:

                    # Save before starting a new stroke
                    history.append(canvas.copy())

                    prev_x, prev_y = x, y

                # Eraser
                if eraser_mode:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        (0, 0, 0),
                        eraser_size
                    )

                # Normal drawing
                else:

                    cv2.line(
                        canvas,
                        (prev_x, prev_y),
                        (x, y),
                        draw_color,
                        brush_size
                    )

                prev_x, prev_y = x, y

    else:
        # Hand disappeared
        prev_x, prev_y = 0, 0

    # ----------------------------------
    # COMBINE CAMERA + DRAWING
    # ----------------------------------
    output = cv2.add(frame, canvas)

    # ----------------------------------
    # STATUS TEXT
    # ----------------------------------
    if eraser_mode:

        cv2.putText(
            output,
            "MODE: ERASER",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 0),
            2
        )

    else:

        cv2.putText(
            output,
            "MODE: DRAW",
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            draw_color,
            2
        )

    # Recording status
    if recording:

        cv2.putText(
            output,
            "RECORDING",
            (width - 180, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        video_writer.write(output)

    # Show project
    cv2.imshow(
        "AI Air Drawing System",
        output
    )

    # ----------------------------------
    # KEYBOARD CONTROLS
    # ----------------------------------
    key = cv2.waitKey(1) & 0xFF

    # Q = Quit
    if key == ord("q"):
        break

    # C = Clear
    elif key == ord("c"):

        history.append(canvas.copy())
        canvas = np.zeros_like(frame)

    # U = Undo
    elif key == ord("u"):

        if len(history) > 0:
            canvas = history.pop()

    # S = Save drawing
    elif key == ord("s"):

        filename = "drawing_" + str(
            int(time.time())
        ) + ".png"

        cv2.imwrite(
            filename,
            canvas
        )

        print("Drawing saved:", filename)

    # R = Start/Stop recording
    elif key == ord("r"):

        if not recording:

            filename = "recording_" + str(
                int(time.time())
            ) + ".mp4"

            fourcc = cv2.VideoWriter_fourcc(
                *"mp4v"
            )

            video_writer = cv2.VideoWriter(
                filename,
                fourcc,
                20.0,
                (width, height)
            )

            recording = True

            print(
                "Recording started:",
                filename
            )

        else:

            recording = False

            if video_writer:
                video_writer.release()

            video_writer = None

            print("Recording stopped")

# ----------------------------------
# CLOSE PROGRAM
# ----------------------------------

if video_writer:
    video_writer.release()

cap.release()

hands.close()

cv2.destroyAllWindows()