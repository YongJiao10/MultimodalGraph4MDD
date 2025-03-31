# **Treatment Prediction for Major Depression**

Yong Jiao, Kanhao Zhao, Xinxu Wei, Nancy B. Carlisle, Corey J. Keller, Desmond J. Oathes, Gregory A. Fonzo, Yu Zhang. [**Deep graph learning of multimodal brain networks defines treatment-predictive signatures in major depression.**](https://doi.org/10.1038/s41380-025-02974-6) *Molecular Psychiatry*, 2025.

<div align=center>
<img src="https://github.com/YongJiao10/MultimodalGraph4MDD/blob/main/img/flowchart.png" width="1000">
</div>

## Dataset Used in This Study
We used the publicly available [EMBARC dataset](https://nda.nih.gov/edit_collection.html?id=2199), which includes multimodal neuroimaging and clinical data from patients with Major Depressive Disorder (MDD).

## Functional Connectivity Augmentation
We developed a functional connectivity (FC) augmentation strategy based on **Common Orthogonal Basis Extraction (COBE)** [1].

<div align=center>
<img src="https://github.com/YongJiao10/MultimodalGraph4MDD/blob/main/img/augmentation.png" width="1000">
</div>

## Multimodal Graph Learning
We implemented the graph neural network in PyTorch 2.1.1 to jointly encode fMRI and EEG brain networks.

## 📖 Reference
[1] Zhou, G., Cichocki, A., Zhang, Y., & Mandic, D. P. (2015). [Group component analysis for multiblock data: Common and individual feature extraction](https://ieeexplore.ieee.org/abstract/document/7310871?casa_token=Cdu6A3mH3IEAAAAA:IzoNtiv3PHed1cKE7foyeXkp0gb2o0St4aSEuiQmaFHYZKPa9YU7iS2_ZY81PImCkEYMg_IAsCI). *IEEE Transactions on Neural Networks and Learning Systems*, **27**(11), 2426–2439.

## 📝 Citation
If you find this code useful, please consider citing our paper:

```bibtex
@article{jiao2025multimodal,
  title={Deep graph learning of multimodal brain networks defines treatment-predictive signatures in major depression},
  author={Jiao, Yong and Zhao, Kanhao and Wei, Xinxu and Carlisle, Nancy B and Keller, Corey J and Oathes, Desmond J and Fonzo, Gregory A and Zhang, Yu},
  journal={Molecular Psychiatry},
  year={2025},
  doi={10.1038/s41380-025-02974-6}
}