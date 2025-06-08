import cv2
import numpy as np

from detection_tools.detector import PeopleDetector


def process_video(input_path: str, output_path: str) -> None:
    """
    Обрабатывает видеофайл: выполняет детекцию людей на каждом кадре и сохраняет результат.

    Загружает входной видеофайл, применяет модель PeopleDetector к каждому кадру,
    отрисовывает прямоугольники и вероятности для каждого обнаруженного человека,
    затем сохраняет результат в указанный выходной файл.

    Args:
        input_path (str): Путь к входному видеофайлу.
        output_path (str): Путь к выходному видеофайлу.
    """
    cap = cv2.VideoCapture(input_path)
    detector = PeopleDetector()

    if not cap.isOpened():
        print(f"❌ Не удалось открыть видеофайл: {input_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        result = detector.detect(frame)

        if result.boxes and result.boxes.xyxy is not None:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = map(int, box)
                label = f"person {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame, label, (x1, max(0, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
                )

        writer.write(frame)

    cap.release()
    writer.release()