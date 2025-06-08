import os
import shutil
import tempfile

from fastapi import FastAPI, File, UploadFile, BackgroundTasks
from fastapi.responses import FileResponse, HTMLResponse

from detection_tools.video_utils import process_video


app = FastAPI()


def remove_file(path: str) -> None:
    """
    Удаляет файл по указанному пути, если он существует.

    Args:
        path (str): Полный путь к файлу.
    """
    if os.path.exists(path):
        os.remove(path)


@app.get("/", response_class=HTMLResponse)
async def home() -> HTMLResponse:
    """
    Возвращает HTML-интерфейс (форму загрузки видео).

    Returns:
        HTMLResponse: HTML-страница с интерфейсом загрузки.
    """
    with open("./templates/index.html", "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())


@app.post("/process/")
async def process_uploaded_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = BackgroundTasks()
) -> FileResponse:
    """
    Обрабатывает загруженное видео:

    - сохраняет во временный файл;
    - запускает процесс детекции;
    - сохраняет обработанный файл;
    - удаляет временные файлы после отправки ответа.

    Args:
        file (UploadFile): Загруженный видеофайл (.mp4).
        background_tasks (BackgroundTasks): Очистка временных файлов.

    Returns:
        FileResponse: Обработанный видеофайл .mp4.
    """
    base_name, ext = os.path.splitext(os.path.basename(file.filename))
    output_filename = f"{base_name}_processed{ext}"

    # Сохраняем входной файл во временный каталог
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp_input:
        shutil.copyfileobj(file.file, temp_input)
        input_path = temp_input.name

    # Путь к выходному видео
    output_path = os.path.join(tempfile.gettempdir(), output_filename)

    # Детекция
    process_video(input_path=input_path, output_path=output_path)

    # Очистка после отправки
    background_tasks.add_task(remove_file, input_path)
    background_tasks.add_task(remove_file, output_path)

    return FileResponse(
        path=output_path,
        media_type="video/mp4",
        background=background_tasks,
        filename=output_filename,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000)