# Проект: Бинарная классификация для предиктивного обслуживания оборудования

## Цель проекта
Разработать модель машинного обучения, которая предсказывает, произойдет ли отказ оборудования (Target = 1) или нет (Target = 0).  
Результаты оформлены в виде интерактивного Streamlit-приложения.

## Описание датасета
**Источник:** [UCI AI4I 2020 Predictive Maintenance Dataset](https://archive.ics.uci.edu/dataset/601/predictive+maintenance+dataset)
- **Объем:** 10,000 записей
- **Признаки:** температура, крутящий момент, скорость вращения, износ и др.
- **Целевая переменная:** `Machine failure` (0 или 1)

## Модель
- **Алгоритм:** Random Forest
- **Метрики оценки:**
  - Accuracy
  - ROC-AUC
  - Confusion Matrix
- **Предобработка:**
  - Удалены лишние признаки (UDI, Product ID, TWF и др.)
  - Кодирование `Type`
  - Масштабирование признаков
## Интерфейс приложения
Приложение позволяет:
- Загружать данные (CSV)
- Обучать модель и визуализировать метрики
- Вводить новые значения и получать предсказание
## Установка и запуск
```bash
https://github.com/misteRamzi/MazitovRR.git
git clone https://github.com/misteRamzi/MazitovRR.git
cd MazitovRR\predictive_maintenance_project
pip install -r requirements.txt
streamlit run app.py
```

## Видео-демонстрация
[Ссылка на видео](predictive_maintenance_project/video/demo.mp4) или встроенное видео ниже:
<video src="predictive_maintenance_project/video/demo.mp4" controls width="100%"></video>
