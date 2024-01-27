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

### Experiment 7-8 (hardy-lake-34/fresh-monkey-28)
Аналогично [Experiment 3](#experiment-3-legendary-sound-13) заморозил всю остальную модель. Использовал multihead attention. Тренировал только голову.

```python
class VideoTextClassif2(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        # чтобы векторы с видео модели, совпали с векторами из языковой
        self.video_projection = torch.nn.Linear(
            1024,
            768,
            bias=False,
        )
        self.multihead_attn = nn.MultiheadAttention(768, 4)

        self.linear = torch.nn.Linear(
            768,
            labels,
        )

    def forward(self, x):
        result = self.model(x)
        language_hidden_state = result["language_encoder"]
        batch_size = language_hidden_state.shape[0]
        frames = 8
        video_hidden_state = result["video_encoder"][:, 0, :]
        video_hidden_state = video_hidden_state.reshape(batch_size, frames, -1)
        video_hidden_state = self.video_projection(video_hidden_state)
        total_hidden_state = torch.cat(
            [video_hidden_state, language_hidden_state],
            dim=1,
        )
        total_hidden_state = language_hidden_state
        attn_output, attn_output_weights = self.multihead_attn(
            total_hidden_state,
            total_hidden_state,
            total_hidden_state,
        )
        feature_vector = attn_output.mean(1)
        result = self.linear(feature_vector)
        return result
```
Результат:
- f1=35.56
- модель показала самый низкий скор на eval loss, однако это никак не помогло ей в целевой метрике
- [wandb-1](https://wandb.ai/dimweb/semeval_emotion_classification/runs/9c9i5n9c?workspace=user-dimweb)
- [wandb-2](https://wandb.ai/dimweb/semeval_emotion_classification/runs/4dqtyzrp?workspace=user-dimweb)
- В ссылках выше я использовал разное количество голов в attention. По итогу это никак не повлияло на результат, только может быть на более скорую сходимость.
- 

### Experiment 9 
Задача: необходимо по реплике предсказать ее эмоцию и сказать какая реплика в диалоге послужила причиной данной эмоции.
Предполагаемое решение: сначала мы берем реплику из диалога и предсказываем ее эмоцию. если эмоция не нейтральная, тогда нам нужно узнать что послужило причиной для нее(или ничего не послужило). Для этого составляем датасет с положительными и отрицательными парами.

Положительная пара: 
```text
(
    реплика с эмоцией,
    реплика, которая послужила причиной для нее
)
```

Чтобы датасет был сбалансированным необходимо в равном количестве предоставвить негативные пары.

Негативные пара:
Составляем всевозможные пары реплик. Выкидываем все положительные пары. Рандомно их перемешиваем и делаем семпл размера положительных пар.
```text
(
    реплика со случайно выбранной эмоцией,
    реплика, которая тоже была выбрана случайно парой для нее
)
```

Одна пара имеет следующий вид(это позитивная):
```python
{
    'initial': {
        'emotion': 'sadness',
        'speaker': 'Monica',
        'text': 'Mr . Heckles .',
        'utterance_ID': 1,
        'video_name': 'dia187utt1.mp4'
    },
    'cause': {
        'emotion': 'sadness',
        'speaker': 'Monica',
        'text': 'Mr . Heckles .',
        'utterance_ID': 1,
        'video_name': 'dia187utt1.mp4'
    },
    'label': 1
}
```

```python
class CauseVideoTextClassif(torch.nn.Module):
    def __init__(self, labels=2, clip_type=None):
        super().__init__()
        self.model = LanguageBind(
            clip_type=clip_type,
            cache_dir="/code/cache_dir",
        )
        self.emotion_classif = torch.nn.Linear(
            768 * 4,
            labels,
        )
        self.cause_classif = torch.nn.Linear(
            768 * 4,
            2,
        )

    def forward(self, x):
        initial_result = self.model(
            {
                "video": x["initial_video"],
                "language": x["initial_language"],
            }
        )
        cause_result = self.model(
            {
                "video": x["cause_video"],
                "language": x["cause_language"],
            }
        )

        features = torch.cat(
            [
                initial_result["video"],
                initial_result["language"],
                cause_result["video"],
                cause_result["language"],
            ],
            dim=-1,
        )
        emotion = self.emotion_classif(features)
        cause = self.cause_classif(features)
        return emotion, cause
```

Результат:
- cause_f1=64.79
- emotion_f1=37.81
- [wandb](https://wandb.ai/dimweb/semeval_cause_classification/runs/trju5u33?workspace=user-dimweb)
- Сетка переобучилась, нужно попробовать оставить только лору.