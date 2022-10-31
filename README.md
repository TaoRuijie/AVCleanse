# Audio-visual speaker recognition

## Introduction

Audio-visual speaker recognition on VoxCeleb2. which includes the speaker recognition code (**speaker** folder), face recognition code (**face** folder) and speaker-face recognition code (**speaker_face** folder). We seperate the codes into three folders to convinent the usage. 

Our paper is [here](https://arxiv.org/pdf/2210.15903.pdf). This project contains the code of **audio-visual speaker recognition** and the [cleansed training list](https://drive.google.com/file/d/1KiilDUZM1lWpo_unkin0qz1xqx0ORKp8/view?usp=share_link). 

This code uses Mixed Precision Training (torch.cuda.amp).

## Preparation

### Requirements
`pip install -r requirements.txt`

### Pretrain model
The link of the pretrain model can be found [here](https://drive.google.com/drive/folders/1W3c6V5bfGZTfwJLJq6ORSXXCLAsG7l2U?usp=share_link).

```
A-Vox2.model:  The speaker network (ECAPA-L) trained on VoxCeleb2
V-Vox2.model:  The face network (ResNet18) trained on VoxCeleb2
V-Glint.model: The face network (ResNet50) trained on Glint360K
```

Create a **pretrain** folder in root directory, put these models into the **pretrain** folder.

### Dataset and text files 

The `.txt` files can be found [here](https://drive.google.com/drive/folders/1yYC38KAaBSr5h0EbMtJ2Vno46rknZDoz?usp=share_link).
The faces on VoxCeleb1 can be downloaded [here](https://www.robots.ox.ac.uk/~vgg/research/CMBiometrics/)
There is no offical link for downloading the videos from [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). So sorry I can not help - -.

The structure of the dataset looks like: 

```
    # VoxCeleb2
    # ├── frame_align (face frames dataset, after alignment)
    # │   ├── id00012 (speaker id)
    # │       ├── 21Uxsk56VDQ (video id)
    # │           ├── 00001 (utterance id)
    # │               ├── 0010.jpg (face frames, I extract one frame every 0.4 second)
    # │               ├── 0020.jpg
    # │               ├── 0030.jpg
    # │   ├── ...
    # ├── wav (speech dataset)
    # │   ├── id00012 (speaker id)
    # │       ├── 21Uxsk56VDQ (video id)
    # │           ├── 00001.wav (utterance id)
    # │   ├── ...
    # ├── train_all.txt (speaker_id-wav_file_name-duraition)
    # ├── train_all_clean.txt (speaker_id-wav_file_name-duraition-audio_sim_score-visual_sim_score-clean_or_noisy)

    # VoxCeleb1
    # ├── frame_align (face frames dataset, be similar to Vox2)
    # ├── wav (speech dataset, be similar to Vox2)
    # ├── O_list.txt (data list of VoxCeleb1-O, wav_file_name-duration)
    # ├── E_list.txt (data list of VoxCeleb1-E)
    # ├── H_list.txt (data list of VoxCeleb1-H)
    # ├── O_trials.txt (original test trials of VoxCeleb1-O)
    # ├── E_trials.txt (original test trials of VoxCeleb1-E)
    # ├── H_trials.txt (original test trials of VoxCeleb1-H)
```

The `O_list`, `E_list` and `H_list` are used to speed up the testing process; `train_all.txt` is the original training list; `train_all_clean.txt` is the cleansed training list.

For face alignment, I do it based on [here](https://github.com/deepinsight/insightface/blob/master/alignment/coordinate_reg/image_infer.py). [Here](https://drive.google.com/file/d/1kYqsimHTNbVlvOkg_nqRzpWbgDC9JqcN/view?usp=sharing) is my code for reference (not write so well -_-...)

## Speaker Recognition

In **speaker** folder, we train the ECAPA-TDNN speaker network on VoxCeleb2. The details of speaker recognition can be found here: [ECAPA-TDNN](https://github.com/TaoRuijie/ECAPA-TDNN). 

| Modality   | System              |  Vox1-O  |  Vox1-E  |  Vox1-H  |
| -------    | ------------        |  ------  |  ------  |  ------  |
|  Speech    | (1) ECAPA-L-Vox2    |   0.98   |  1.21    |  2.30    |

It is noted that the results in our paper are the mean performance of training three times, so be slightly different with these results.

## Face Recognition

In **face** folder, we train a face recognition model on VoxCeleb2, here are the results:

| Modality   | System              |  Vox1-O  |  Vox1-E  |  Vox1-H  |
| -------    | ------------        |  ------  |  ------  |  ------  |
|  Face      | (2) ResNet18-Vox2   |   0.97   |  0.81    |  1.16    |
|  Face      | (3) ResNet50-Glint  |   0.03   |  0.07    |  0.09    |

It is noted that (3) is a [pretrain-model](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) on the Glint360K dataset. We did not check if there is the identity overlap between Glint360K (360K person) and VoxCeleb (6K person) set (lack the identity files). This result is only used to show that the multi-modality is strong, but that will not effect our cleansing purpose since our final target is a cleansed VoxCeleb2.

We did not do experiments that had no alignments for both training and evaluation. For the pretrain face model on Glint360K, the result is bad if no alignment for evaluation.

## Speaker-Face Recognition

The pipeline of face recognition is in **speaker_face** folder. We suggest to train two networks separately. Here are the results:

| Modality   | System              |  Vox1-O  |  Vox1-E  |  Vox1-H  |
| -------    | ------------        |  ------  |  ------  |  ------  |
|  Fusion    | (1)+(2)             |   0.15   |  0.23    |  0.41    |
|  Fusion    | (1)+(3)             |   0.01   |  0.08    |  0.15    |

## Training and evaluation

Get into the `speaker`, `face` or `speaker_face` folder. 

For training: `bash run_train.sh`

Noted to set the path of training data, evaluation data, and the output path.

For evaluation: `bash run_eval.sh`.

Noted to set the path of evaluation data.

I have optimized the code to speed up the evalution process (': mins, '': seconds)

| Modality   | System              |  Vox1-O  |  Vox1-E  |  Vox1-H  |
| -------    | ------------        |  ------  |  ------  |  ------  |
|  Speech    | (1)                 |   **0'14''** |  **5'30''**  |  **5'16''**  |
|  Face      | (2)                 |   0'13'' |  5.33''  |  5'30''  |
|  Face      | (3)                 |   0'24'' |  11‘26’‘ |  10'52'' |
|  Fusion    | (1)+(2)             |   0'25'' |  11'26'' |  10'55'' |
|  Fusion    | (1)+(3)             |   0'38'' |  16‘57’‘ |  16‘11’‘ |

For speech modality only, we can evaluate Vox1-E and Vox1-H within 6 mins. (One RTX-3090)
