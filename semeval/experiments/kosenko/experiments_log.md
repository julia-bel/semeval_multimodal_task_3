### Experiment 1
Задача: построить классификатор эмоций в диалоге на основе видео, которое соответствует данному отрывку. 
Описание эксперимента: Взять в качестве эмбеддера languagebind. Затем использовать сконкатенированные репрезентации текста и видео в качестве фич для классификатора. Отрывки текста с видео берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

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
- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/29ddd1e478f48bf30886e341e8c0e2eab53dd8bd/semeval/experiments/kosenko/language_bind/languagebind_classification.py)

- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/tjy9q8sz?workspace=user-dimweb)
- [weights link](https://huggingface.co/dim/SemEvalParticipants_models/tree/main/kosenko_exp_1)

Максимальный **f1_score 0.3646**. Согласно графику обучения видно что модель переобучилась, причем очень сильно. Для снижения данного эффекта можно применить lora. Теперь нужно попробовать сделать тоже самое, плюс добавить аудио.

### Experiment 2
Задача: построить классификатор эмоций в диалоге на основе аудио и видео, которое соответствует данному отрывку.
Описание эксперимента: Взять в качестве эмбеддера languagebind. Затем использовать сконкатенированные репрезентации текста и видео, аудио в качестве фич для классификатора. Отрывки текста с видео, аудио берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

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

Единственное отличие от [Experiment 1](#experiment-1) это модель.

```python
class VideoAudioTextClassif(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.linear = torch.nn.Linear(
            768 * 3,
            labels,
            bias=False,
        )

    def forward(self, x):
        result = self.model(x)
        # print(result)
        features = torch.cat(
            [
                result["video"],
                result["audio"],
                result["language"],
            ],
            dim=-1,
        )
        result = self.linear(features)
        return result
```
Результат: