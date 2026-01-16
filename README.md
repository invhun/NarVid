# Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions

The official implementation of NarVid — a framework that enhances text-video retrieval by leveraging frame-level captions (narration) to improve semantic understanding and retrieval accuracy. NarVid employs cross-modal interactions, query-aware filtering, dual-modal matching, and hard-negative loss, achieving SOTA results on MSR-VTT, MSVD, VATEX and DiDeMo.
![framework](/docs/framework.png)

## News
[Feb 27, 2025] Our paper has been accepted to CVPR 2025

## Requirements
This project requires two environments: one for `NarVid`, which serves as the main framework, 
and one for `LLaVa`, which is used for preprocessing to generate narrations.

### Setting up the Main NarVid Environment
```sh
conda install --yes -c pytorch pytorch=1.13.1 torchvision cudatoolkit=11.6
pip install opencv-python==4.9.0.80 numpy==1.23.0 ftfy regex tqdm boto3 requests pandas
```

### Setting up the LLaVa Environment

For setting up the llava environment, please refer to the official GitHub repository: [LLaVA](https://github.com/haotian-liu/LLaVA/tree/main).

## Data Preparation

### For MSRVTT

Raw videos can be download from [link](https://cove.thecvf.com/datasets/839)
The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msrvtt)

### For MSVD

Raw videos can be download from [link](https://www.cs.utexas.edu/~ml/clamp/videoDescription/)
The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/tree/master/misc/datasets/msvd)

### For VATEX

Raw Videos and split can be download from [Vatex](https://eric-xw.github.io/vatex-website/download.html) 

### For DiDeMo

Raw videos can be download from [LisaAnne/LocalizingMoments.](https://github.com/LisaAnne/LocalizingMoments). The splits can be found in the job [collaborative-experts](https://github.com/albanie/collaborative-experts/blob/master/misc/datasets/didemo/README.md).


## Data Preprocessing

For convenient reproduction of our research, we provide both data preprocessing scripts and pre-generated narration files.

### Extract Video Frames

Before generating captions for each frame, you need to perform preprocessing on the raw video to extract the frames.

```sh
python preprocess/video_frame_extractor.py --raw_video_folder [your_raw_video_folder_path] --extracted_frame_path [your_output_frame_path]
```

### Generate Narration from Frames

Based on the extracted video frames, use LLaVa to generate captions for each frame.

```sh
python preprocess/narration/narration_generator.py --video_frames_path [your_frame_path] --video_id_list_path [your_video_id.json]
```

### Download Pre-generated Narration Files

To simplify reproduction, pre-generated narration files are available for direct download. These files include narrations for the MSR-VTT, DiDeMo, MSVD, and VATEX datasets, generated using the above process.

*   **[NarVid v1.1 Release Page](https://github.com/invhun/NarVid/releases/tag/1.1)**



## How to Run 

（1）Ensure that the data preparation and preprocessing are completed

You can check the narration for the MSR-VTT, MSVD, and VATEX datasets in the `narration_data` directory. Please note that the Didemo dataset is not included due to file size limitations

（2）About the pretrained CLIP checkpoints 

Download CLIP (ViT-B/32) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/40d365715913c9da98579312b702a82c18be219cc2a73407c4526f58eba950af/ViT-B-32.pt
```
or, download CLIP (ViT-B/16) weight,
```sh
wget -P ./modules https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt
```

（3）About the running scripts

### MSRVTT

```sh
sh scripts/train_msrvtt_vit32.sh
```

```sh
sh scripts/train_msrvtt_vit16.sh
```

### MSVD

```sh
sh scripts/train_msvd.sh
```

### VATEX

```sh
sh scripts/train_vatex.sh
```

### DiDeMo

```sh
sh scripts/train_didemo.sh
```

# Citation
If you find NarVid helpful for your work, please cite the following paper when using our code or referring to the results.
```bibtex
@InProceedings{Hur_2025_CVPR,
    author    = {Hur, Chan and Hong, Jeong-hun and Lee, Dong-hun and Kang, Dabin and Myeong, Semin and Park, Sang-hyo and Park, Hyeyoung},
    title     = {Narrating the Video: Boosting Text-Video Retrieval via Comprehensive Utilization of Frame-Level Captions},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2025},
    pages     = {24077-24086}
}
```

# Acknowledgments
The implementation of NarVid relies on resources from [CLIP](https://github.com/openai/CLIP "CLIP"), [CLIP4Clip](https://github.com/ArrowLuo/CLIP4Clip "CLIP4Clip") and [Cap4Video](https://github.com/whwu95/Cap4Video "Cap4Video").
