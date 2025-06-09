from ultralytics import YOLO
import numpy as np


class PeopleDetector:
    """
    Детектор людей с использованием модели YOLO.

    Атрибуты:
        model (YOLO): Предобученная модель YOLO.
        conf_threshold (float): Порог уверенности для фильтрации предсказаний.
        person_class_id (int): ID класса "человек" в датасете COCO.
    """

    def __init__(self, model_path: str = "yolo11m.pt"):
        """
        Инициализирует PeopleDetector с заданным путём к весам модели.

        Args:
            model_path (str): Путь к .pt-файлу модели YOLO.
        """
        self.model = YOLO(model_path)
        self.conf_threshold = 0.05
        self.person_class_id = 0  # класс "человек" в COCO

    def detect(self, frame: np.ndarray):
        """
        Выполняет детекцию людей на одном кадре.

        Args:
            frame (np.ndarray): Изображение в формате BGR (например, из OpenCV).

        Returns:
            Result: Объект первого результата инференса модели (ultralytics.engine.results.Results).
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf_threshold,
            classes=[self.person_class_id],
            imgsz=1280,
            agnostic_nms=True,
            verbose=False
        )
        return results[0]
