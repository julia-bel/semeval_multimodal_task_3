## Классификация эмоций
[Wandb link](https://api.wandb.ai/links/julia-bel/ktcy5ewd)

### Experiment 0 (multimodal_classification_0)
- **f1_score=0.2372**
- [notebook](./videollama/pybooks/emotion_classification_experiment_0.ipynb)

**Задача**: построить классификатор эмоций в беседе на основе аудио, видео и текста, которые соответствуют некоторому отрывку беседы.
**Описание эксперимента**: взять в качестве эмбеддера видео и аудио выделенный из VideoLLaMA backbone и текстовые эмбеддинги VideoLLaMA. Архитектура клласификатора: совместить эмбеддинги модальностей: уменьшить их размерности, применить пулинг средним, сконкатенировать. Далее применить линейную голову классификатора. Эмбеддинги не обучаются.  
**Примечание**: отрывки берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

Один батч для классификации выглядит следующим образом: 
```python
{
    "video": (batch_size, channels, frames, height, width),
    "text":  (batch_size, max_text_len),
    "audio": (batch_size, frames, channels, height, width),
    "label": (batch_size,),
}
```

Ключевые особенности архитектуры:
```python
def __init__(
        self,
        video_embedding,
        audio_embedding,
        text_embedding,
        input_dim=5120,
        hidden_dim=1024,
        num_classes=7,
    ):
    ...
    self.video_embedding = video_embedding
    self.audio_embedding = audio_embedding
    self.text_embedding = text_embedding
    self.projections = nn.ModuleList(
        [nn.Linear(input_dim, hidden_dim) for _ in range(3)]
    )

    self.classifier = nn.Sequential(
        nn.Linear(hidden_dim * 3, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, num_classes)
    )
    ...

def forward(self, text, video, audio):
    embeddings = [
        self.text_embedding(text),
        self.video_embedding(video),
        self.audio_embedding(audio),
    ]
    projections = [
        torch.mean(p(e.float()), dim=1) 
        for p, e in zip(self.projections, embeddings)
    ]
    concat_features = torch.cat(projections, dim=1)
    logits = self.classifier(concat_features)
    return logits
```

**Результаты**  
Согласно графику обучения видно, что модель имеет крайне слабую обобщающую способность. Основное предположение состоит в том, что пулинг средним в данном случае не подходит, так как векторы имеют слишком сложную структуру, а также последовательности могут содержать значительный размер паддинга.

### Experiment 1 (multimodal_classification_1)
- **f1_score=0.4307**
- [notebook](./videollama/pybooks/emotion_classification_experiment_1.ipynb)

**Задача**: построить классификатор эмоций в беседе на основе аудио, видео и текста, которые соответствуют некоторому отрывку беседы.
**Описание эксперимента**: в данном эксперименте используются только две модальности: видео и текст (далее планируется добавить аудио). Эксперимент развивает идею Experiment 0, заменяя пулинг средним на механизм внимания и уменьшая размер скрытого представления. Эмбеддинги не обучаются.
**Примечание**: отрывки берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

Ключевые особенности архитектуры:
```python
def __init__(
        self,
        video_embedding,
        text_embedding,
        input_dim=5120,
        hidden_dim=512,
        attention_dim=128,
        num_classes=7,
    ):
    ...
    self.video_embedding = video_embedding
    self.text_embedding = text_embedding

    self.projections = nn.ModuleList(
        [nn.Linear(input_dim, hidden_dim) for _ in range(2)]
    )
    self.attention_layers = nn.ModuleList(
        [Attention(hidden_dim, attention_dim) for _ in range(2)]
    )

    self.classifier = nn.Sequential(
        nn.Linear(hidden_dim * 2, hidden_dim),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(hidden_dim, num_classes)
    )
    ...

def forward(self, text, video):
    embeddings = [
        self.text_embedding(text),
        self.video_embedding(video),
    ]
    projections = [
        att(p(e.float()))
        for att, p, e in zip(self.attention_layers, self.projections, embeddings)
    ]
    concat_features = torch.cat(projections, dim=1)
    logits = self.classifier(concat_features)
    return logits
```

**Результаты**  
Данная модель имеет на текущий момент наилучшее качество. Однако количество эпох необходимо увеличить и дообучить, так как плато явно не достигнуто.

### Experiment 3 (multimodal_classification_3)
- **f1_score=0.4139**

**Задача**: построить классификатор эмоций в беседе на основе аудио, видео и текста, которые соответствуют некоторому отрывку беседы.
**Описание эксперимента**: эксперимент аналогичен Experiment 1, но используются все три модальности. 
**Примечание**: отрывки берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

**Результаты**  
Сравнимое качество, обучение менее стабильное.

### Experiment 4 (multimodal_classification_4)
- **f1_score=0.3941**

**Задача**: построить классификатор эмоций в беседе на основе аудио, видео и текста, которые соответствуют некоторому отрывку беседы.
**Описание эксперимента**: эксперимент аналогичен Experiment 1, но дополнительно обучаются эмбеддинги текста.
**Примечание**: отрывки берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

**Результаты**  
Качество на тренировочной выборке хорошее, однако на валидации качество низкое. Происходит переобучение.

### Experiment 3 (multimodal_classification_3)
- **f1_score=0.4139**

**Задача**: построить классификатор эмоций в беседе на основе аудио, видео и текста, которые соответствуют некоторому отрывку беседы.
**Описание эксперимента**: эксперимент аналогичен Experiment 1, но используются все три модальности. 
**Примечание**: отрывки берутся из диалога независимо. Модель не имеет информации о их взаимосвязи и порядке.

**Результаты**  
Сравнимое качество, обучение менее стабильное.

## Совместная классификация
[Wandb link](https://api.wandb.ai/links/julia-bel/echjbr5m)

### Experiment 0 (joint_classification_0)
- **emotion: f1_score=0.3341**
- **causal: f1_score=0.3325**
- [notebook](./videollama/pybooks/joint_classification_experiment_0.ipynb)

**Задача**: построить классификатор эмоций и их причин в беседе на основе аудио, видео и текста.  
**Описание эксперимента**: реализован функционал совместной тренировки двух классификаторов. Эмоции классифицируются по независимым фрагментам беседы, а для классификации причин используются логиты эмоций отдельных фрагментов беседы и каждая беседа целиком. Для представления причинно-временной связи в беседе используются bilstm. Другие детали архитектуры приведены в ноутбуке.

**Результаты**  
Совместное качество двух классификаторов довольно низкое. Однако в целом график обучения достаточно неплохой. Можно усложнить архитектуру классификаторов, оптимизировать гиперпараметры и тд. Или использовать другие эмбеддеры.
