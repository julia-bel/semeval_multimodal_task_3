

# Dataset for SemEval-2024 Task 3

The dataset for [SemEval-2024 Task 3: The Competition of Multimodal Emotion Cause Analysis in Conversations](https://nustm.github.io/SemEval-2024_ECAC/) is released here.



## File Description

- The `text` folder contains the files of training data for each subtask, where all instances are stacked into a list and each of them is stored in a dictionary.
- The `video_with_audio` folder contains the video files required for Subtask 2. Note: The three compressed packages correspond to the training, validation, and testing sets of ECF respectively. All of them are used as the training set in this SemEval evaluation.

```
SemEval-2024_Task3
├── text
│   ├── Subtask_1_train.json
│   ├── Subtask_2_train.json
├── video_with_audio
│   ├── train_1.tar.gz
│   │   ├── dia1utt1.mp4
│   │   ├── dia1utt2.mp4
│   │   ├── ...
│   ├── train_2.tar.gz
│   ├── train_3.tar.gz
├── README.md
```



## Dataset Statistics

| # of Instances (Training Data)            | # of Instances (Evaluation data) |
| ----------------------------------------- | -------------------------------- |
| 1,374 conversations <br>13,619 utterances | To be updated.                   |



