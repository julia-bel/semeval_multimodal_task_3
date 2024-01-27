### Experiment 1 (dauntless-tree-10)
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

### Experiment 2 (woven-microwave-12)
Задача: построить классификатор эмоций в диалоге на основе аудио и видео, текста который соответствует данному отрывку.
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

Единственное отличие от [Experiment 1](#experiment-1-dauntless-tree-10) это модель.

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

Из-за увеличения количества параметров, сеть быстро переобучается. На общем созвоне предложили заморозить все модели и оставить только линейные классификаторы.
- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/251a3a090b19351c93f7ff5d7d301fb5060e58b5/semeval/experiments/kosenko/language_bind/languagebind_classification_video_audio_text.py)

- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/824rfp0q?workspace=user-dimweb)
- f1_score=0.3201

### Experiment 3 (legendary-sound-13)
Аналогично [experiment_1](#experiment-1-dauntless-tree-10). В данном эксперименте я замораживаю всю модель и оставляю только голову.

Результат:
f1_score еще ниже чем, при полном файнтюне. Судя по трейн лоссу, теперь модели не хватает параметров. Следует попробовать lora.

- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/01708a7c0cc1219136272c9d26cae5905532c989/semeval/experiments/kosenko/language_bind/languagebind_classification_video_text.py)
- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/55qtfuxr?workspace=user-dimweb)


### Experiment 4 (soft-field-14)
Аналогично [experiment_1](#experiment-1-dauntless-tree-10). В данном эксперименте я обучаю только лора слои. На классификатор лора слой не добавляю. Лора слои добавляются только на attention механизм.

Конфиг для лора.
```python
peft_config = LoraConfig(
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="all",
    target_modules=[
        'k_proj',
        'v_proj',
        'q_proj',
        'out_proj',
    ]
)

```
- trainable params: 7,968,512 || all params: 536,105,728 || trainable%: 1.486369494638192

Результат:
Результат намного лучше чем с простой головой, но видно что модели не хватает параметров. Даже через 10 эпох она не переобучилась. Необходимо добавить lora к большему количеству слоёв.
- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/636c5503d504a864b91bda88000754a34339b66f/semeval/experiments/kosenko/language_bind/languagebind_classification_video_text.py)
- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/zma75ds9?workspace=user-dimweb)



### Experiment 5 (cerulean-field-15)
Аналогично [experiment_1](#experiment-4). В данном эксперименте я увеличиваю количество слоев, к которым применяется lora.

Конфиг для лора.
```python
peft_config = LoraConfig(
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="all",
    target_modules=[
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj",
        "fc1",
        "fc2",
    ],
)
```
- trainable params: 13,375,232 || all params: 541,512,448 || trainable%: 2.469976830523386
Результат:
Увеличение количества параметров улучшило сходимость и скор по f1_score. Но это порежнему очень мало, в сравнении с полным файнтюном. По графикам обучения и эвалюации видно что теперь не хватает обобщающей способности. Необходимо задействовать аудио с текущим конфигом lora.

- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/e32f2350efbfc8dcb4dd0ed2e4930e374bf1dfec/semeval/experiments/kosenko/language_bind/languagebind_classification_video_text.py)
- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/5a1v9oep?workspace=user-dimweb)

### Experiment 6 (gallant-wind-27)
Аналогично [experiment 2](#experiment-2-woven-microwave-12). Обучаю только lora со следующим конфигом.
```python
peft_config = LoraConfig(
    inference_mode=False,
    r=16,
    lora_alpha=16,
    lora_dropout=0.0,
    bias="all",
    target_modules=[
        "k_proj",
        "v_proj",
        "q_proj",
        "out_proj",
        "fc1",
        "fc2",
    ],
)
```
- trainable params: 20,725,504 || all params: 852,905,984 || trainable%: 2.429986937458279

Результат:
- f1_score=0.3375
- сеть не переобучилась, как в прошлый раз, но и не показала выдающегося результата. Это печально.
- [commit](https://github.com/julia-bel/SemEvalParticipants/blob/33ba9ea6db522e46edd0f7bdb8aec2d4af455a2f/semeval/experiments/kosenko/language_bind/languagebind_classification_video_audio_text.py)
- [wandb link](https://wandb.ai/dimweb/semeval_emotion_classification/runs/jqegel3m?workspace=user-dimweb)
