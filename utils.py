import cv2

import numpy as np
from ultralytics import YOLO, solutions


def detect_objects_with_video(video_path,
                              model_path,
                              output_result_path,
                              width=1200,
                              height=900):

    model = YOLO(model_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60
    frame_size = (width, height)

    out = cv2.VideoWriter(output_result_path, fourcc, fps, frame_size)

    while True:

        ret, frame = cap.read()

        if not ret:
            print("End of video or error reading frame.")
            break

        result = model(frame, imgsz=640, verbose=False)[0]

        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy.cpu().numpy()[0])
            label = result.names[int(box.cls.cpu().numpy()[0])]
            confidence = box.conf.cpu().numpy()[0]

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            label_text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

        resized_frame = cv2.resize(frame, frame_size)

        out.write(resized_frame)

        cv2.imshow("Video Frame", resized_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print(f"Video saved as {output_result_path}")


def detect_objects_and_speed_with_video(video_path,
                                        model_path,
                                        output_result_path,
                                        speed_region,
                                        width=1200,
                                        height=900):

    cap = cv2.VideoCapture(video_path)
    assert cap.isOpened(), "Error reading video file"
    w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                 cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 60

    video_writer = cv2.VideoWriter(
        output_result_path, fourcc, fps, (width, height))

    speed = solutions.SpeedEstimator(
        show=False,
        model=model_path,
        region=speed_region,
    )

    while cap.isOpened():
        success, im0 = cap.read()
        if not success:
            print(
                "Video frame is empty or video processing has been successfully completed.")
            break

        out = speed.estimate_speed(im0)

        resized = cv2.resize(im0, (width, height))

        video_writer.write(resized)

        cv2.imshow("Processed Video", resized)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Processing stopped by user.")
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
