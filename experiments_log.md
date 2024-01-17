### Experiment 1
Задача: построить классификатор эмоций в диалоге на основе видео, которое соответствует данному отрывку. 
Описание эксперимента: Взять в качестве эмбеддера languagebind. Затем использовать сконкатенированные репрезентации текста и аудио в качестве фич для классификатора. Отрывки текста с видео берутся из диалога независимо. Модель не имеет о их взаимосвязи и порядке.

Один батч для классификации выглядит следующим образом: 
```python
{
	'text': ['Th ... th ... that is all it is , a third nipple .', 'What ? !'],
	'video_name': ['/code/SemEval-2024_Task3/training_data/train/dia424utt1.mp4',
	'/code/SemEval-2024_Task3/training_data/train/dia1255utt10.mp4'],
	'label': tensor([3, 0])
}
```
Для того чтобы преобразовать лейблы в текст, используется следующий мап.
```python
all_emotions = [
        "surprise",
        "fear",
        "sadness",
        "neutral",
        "joy",
        "anger",
        "disgust",
    ]
emotions2labels = {em: i for i, em in enumerate(all_emotions)}
labels2emotions = {i: em for i, em in enumerate(all_emotions)}
```

Параметры обучения такие:
```python
training_args = TrainingArguments(
	output_dir="semeval/experiments/kosenko/language_bind/train_results/",
	evaluation_strategy="epoch",
	num_train_epochs=10,
	save_strategy="epoch",
	save_total_limit=1,
	report_to="wandb",
	logging_steps=5,
	per_device_train_batch_size=4,
	per_device_eval_batch_size=4,
	gradient_accumulation_steps=16,
	bf16=True,
	remove_unused_columns=False,
)
```
Результат: 