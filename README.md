Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing (To appear in ECCV 2020) [[Paper]](https://arxiv.org/pdf/2007.10558.pdf)

[Yapeng Tian](http://yapengtian.org/), [Dingzeyu Li](https://dingzeyu.li/), and [Chenliang Xu](https://www.cs.rochester.edu/~cxu22/) 

### Audio-visual video parsing

We define the <b>Audio-Visual Video Parsing</b> as a task to group video segments
and parse a video into different temporal audio, visual, and audio-visual events
associated with semantic labels.

![image](Figs/avvp_fig.png)


### LLP Dataset & Features
```bash
# annotations for LLP dataset 
cd data
AVVP_dataset_full.csv: full dataset with weak annotaions
AVVP_train.csv: training set with weak annotaions
AVVP_val_pd.csv: val set with weak annotaionsa
AVVP_test_pd.csv: test set with weak annotaions
AVVP_eval_audio.csv: dense audio event annotations for videos in val and test sets
AVVP_eval_visual.csv: dense visual event annotations for videos in val and test sets
```
Note that audio-visual events can be derived from audio and visual events.

We use [VGGish](https://github.com/tensorflow/models/tree/master/research/audioset/vggish), [ResNet152](https://pytorch.org/docs/stable/torchvision/models.html), and [ResNet (2+1)D](https://pytorch.org/docs/stable/torchvision/models.html) to extract audio, 2D frame-level, and 3D snippet-level features, respectively. 
The audio and visual features of videos in the LLP dataset can be download from this Google Drive [link](). The features are in the "feats" folder.


### Requirements

```bash
pip install -r requirements
```

### Weakly supervised audio-visual video parsing 

Testing: 


```bash
python main_avvp.py --mode train --gpu 0 --audio_dir /xx/feats/vggish/ --video_dir /xx/feats/res152/ --st_dir /xx/feats/r2plus1d_18/
```

Training:

```bash
python main_avvp.py --mode test --gpu 0 --audio_dir /xx/feats/vggish/ --video_dir /xx/feats/res152/ --st_dir /xx/feats/r2plus1d_18/
```
### Download videos (coming soon)

download raw videos
```bash
python main_avvp.py --mode test --gpu 0 --audio_dir /xx/feats/vggish/ --video_dir /xx/feats/res152/ --st_dir /xx/feats/r2plus1d_18/
```

### Feature extraction (coming soon)

extract your own audio and visual features

### Citation

If you find this work useful, please consider citing it.

<pre><code>@InProceedings{tian2018ave,
  author={Yapeng Tian, Dingzeyu Li, and Chenliang Xu},
  title={Unified Multisensory Perception: Weakly-Supervised Audio-Visual Video Parsing},
  booktitle = {ECCV},
  year = {2020}
}
</code></pre>

### License
This project is released under the [GNU General Public License v3.0](https://github.com/Mukosame/Zooming-Slow-Mo-CVPR-2020/blob/master/LICENSE).




