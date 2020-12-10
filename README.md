<h1> Deep Shells: Unsupervised Shape Correspondence with Optimal Transport </h1>

![](https://github.com/marvin-eisenberger/deep-shells/blob/master/teaser_deep_shells.png)

Implementation of the NeurIPS 2020 paper ([arXiv](https://arxiv.org/abs/2010.15261)). For a given pair of deformable 3D shapes, our algorithm produces high-quality correspondences.

## Usage

### Preprocessing
* To preprocess the raw datasets, run the matlab file:
```bash
preprocess_data/preprocess_dataset.m
```

### Datasets
* In our experiments, we use the FAUST remeshed and SCAPE remeshed benchmarks. Both datasets can be downloaded from [here](https://github.com/LIX-shape-analysis/GeomFmaps).
* Change dataset paths under the `get_faustremeshed_file()` and `get_scaperemeshed_file()` functions in `data.py`. 
* To jointly train multiple datasets (including inter-dataset pairs), create a hybrid dataset, see e.g. `FaustScapeRemeshedTrain` in `data.py`. In the paper, we show an experiment where we jointly train on FAUST remeshed and SCAPE remeshed (see Table 1).

### Models
* Checkpoints are saved under:
```bash
./models/
```
* The checkpoint in ./models/Faust_Scape/ corresponds to our results in the 5th and 6th column of Table 1.

### Train
* For FAUST remeshed, run `train_faustremeshed_train()` from `main.py`.
* For SCAPE remeshed, run `train_scaperemeshed_train()` from `main.py`.

### Test
* To output test correspondences for one sample pair, see `demo_faust_scape()` in `main.py`.
* Run your own examples analogously.

### Citation
If you use our implementation, please cite:
```
@article{eisenberger2020deep,
  title={Deep Shells: Unsupervised Shape Correspondence with Optimal Transport},
  author={Eisenberger, Marvin and Toker, Aysim and Leal-Taix{\'e}, Laura and Cremers, Daniel},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  year={2020}
}
```
### Axiomatic matching (smooth shells)
* The repo also contains a vanilla implementation of the CVPR 2020 [paper](https://openaccess.thecvf.com/content_CVPR_2020/papers/Eisenberger_Smooth_Shells_Multi-Scale_Shape_Registration_With_Functional_Maps_CVPR_2020_paper.pdf) smooth shells (the [original implementation](https://github.com/marvin-eisenberger/smooth-shells) is in matlab).
* If you use this part of our implementation, cite the smooth shells paper:
```
@inproceedings{eisenberger2020smooth,
  title={Smooth Shells: Multi-Scale Shape Registration with Functional Maps},
  author={Eisenberger, Marvin and Lahner, Zorah and Cremers, Daniel},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={12265--12274},
  year={2020}
}
```

