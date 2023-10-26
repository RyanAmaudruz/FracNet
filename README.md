# ROBUST FRACNET: LEARNING TO DETECT RIB FRACTURES WITH NOISE





## Abstract
This paper focuses on the segmentation of rib fractures from CT scans, a crucial task for accurate diagnosis and timely intervention in cases of trauma-related injuries. We build upon the FracNet [1] model using various augmentation techniques, namely adding Gaussian noise and mirroring, and sampling methods such as positive and negative sampling. Combination of these methods yields an improvement of 15% over the baseline in terms of the Free-Response Receiver Operating Characteristic (FROC) score and  32.1% in terms of the Dice Score, revealing the potential of these inexpensive methods. Potential future research could investigate the effectiveness of integrating diffusion-based techniques to further enhance the extraction of semantic information from the CT scans.

## Introduction
The past years have witnessed significant advancements in the field of medical imaging, with deep learning and computer vision being at the forefront [2, 3]. The ultimate goal of such technologies is to improve the diagnostic precision of medical imaging and automatize the diagnosis process, which allows professional radiologists to focus on other duties.

One specific application of medical imaging is rib fracture detection, a frequent injury resulting from various traumatic events such as accidents, falls, or sports-related incidents. In the case of rib fractures, the most common imaging modality is chest computed tomography (CT) [4], which provides a detailed representation of the ribs and the surrounding area. Despite this detail, the misdiagnosis rate for rib fractures from CT scans has a misdiagnosis rate of between 19.2% to 26.8% [5, 6] which has serious implications for patients' well-being. Indeed, mortality rates tend to increase with the number of rib fractures [4], underscoring the need for accurate and early detection of rib fractures. This paper therefore focuses on rib fracture detection, particularly rib fracture segmentation.

Most of the advancements in rib fracture detection and segmentation using deep learning models have come from changing model architectures or using more data. The first one is quite costly to develop and deploy, and rib fracture data are hard to acquire because of privacy and ethical considerations, as outlined by Wu et al. [7]. To the best of our knowledge, there is no research focused on data augmentation and sampling methodologies in the context of image segmentation and how they contribute to model robustness, prevent overfitting and improve generalisation. This paper therefore aims to answer the following:

- [RQ1]: How do combined techniques of variable Gaussian noise and mirroring along different axes influence the robustness, generalisability, and orientation invariance of FracNet in the context of rib fracture segmentation in chest CT scans?

- [RQ2]: How does introducing variability in the selection of positive and negative regions of interest, through additive Gaussian noise to centroids and intensity-based criteria, influence the training efficacy and overfitting prevention of FracNet in rib fracture segmentation?

Full paper available upon request. 

Authors: *Amaudruz R., Kapralova M., Palfi B., Pantea L., Turcu A*

## Code Structure
* FracNet/
    * [`dataset/`](./dataset): PyTorch dataset and transforms
    * [`models/`](./models): PyTorch 3D UNet model and losses
    * [`utils/`](./utils): Utility functions
    * [`main.py`](main.py): Training script
    * [`predict.py`](predict.py): Inference script
    * [`rib_fracture_viz.ipynb`](rib_fracture_viz.ipynb): Visualisation notebook

Below is the positive sampling from the original repo (https://github.com/M3DV/FracNet):

![Alt Text](visualisations/ori_pos_sampling_viz-axial.gif)

The fracture is displayed in <font color="red">red</font> and the positive sample fed to the model in <font color="blue">blue</font>.
Below is the effect of the new positive sampling strategy:

![Alt Text](visualisations/mod_pos_sampling_viz-axial.gif)


More visualisations and data exploring widgets are available in the [`rib_fracture_viz.ipynb`] notebook. 


## Usage

### Install Required Packages

To install the main repo environment, run the following in command line:
```bash
conda env create -f environment_gpu.yml
```
To install the visualisation environment, run the following in command line:
```bash
conda env create -f fracnet_viz_environment.yml
```



### Download the Dataset
To run the files, you will need to download the rib fracture dataset here: [RibFrac Challenge](https://ribfrac.grand-challenge.org/dataset/).

<details>
<summary>
Data folder organisation
</summary>

```bash
data/
    └──train/
        ├── ribfrac-train-images/
            ├── RibFrac1-image.nii.gz
            ├── RibFrac2-image.nii.gz
            └── ...
        └── ribfrac-train-labels/
            ├── RibFrac1-label.nii.gz
            ├── RibFrac2-label.nii.gz
            └── ...
    └──val/
        ├── ribfrac-val-images/
            ├── RibFrac421-image.nii.gz
            ├── RibFrac422-image.nii.gz
            └── ...
        └── ribfrac-val-labels/
            ├── RibFrac421-label.nii.gz
            ├── RibFrac422-label.nii.gz
            └── ...
    └──test/
        ├── ribfrac-test-images/
            ├── RibFrac501-image.nii.gz
            ├── RibFrac502-image.nii.gz
            └── ...
```
</details>


### Training
To train the FracNet model, run the following in command line:
```bash
python main
```

### Prediction
To generate prediction, run the following in command line:
```bash
python predict 
```

***Note***: This repo is a fork of https://github.com/M3DV/FracNet.


## References

[1] L. Jin, J. Yang, K. Kuang, B. Ni, Y. Gao, Y. Sun, P. Gao, W. Ma, M. Tan, H. Kang, J. Chen, and M. Li, “Deep-learning-assisted detection and segmentation of rib fractures from ct scans: Development and validation of fracnet” EBioMedicine, 2020.

[2] Y. Zhan, Y. Wang, W. Zhang, B. Ying, and C. Wang, “Diagnostic accuracy of the artificial intelligence methods in medical imaging for pulmonary tuberculosis: A systematic review and meta-analysis” Journal of Clinical Medicine, vol. 12, no. 1, 2023.[Online]. Available: https://www.mdpi.com/2077-0383/12/1/303

[3] R. Aggarwal, V. Sounderajah, G. Martin, D. S. Ting, A. Karthikesalingam, D. King, H. Ashrafian, and A. Darzi, “Diagnostic accuracy of deep learning in medical imaging: a systematic review and meta-analysis” NPJ digital medicine, vol. 4, no. 1, p. 65, 2021.

[4] B. S. Talbot, C. P. Gange Jr, A. Chaturvedi, N. Klionsky, S. K. Hobbs, and A. Chaturvedi, “Traumatic rib injury: patterns, imaging pitfalls, complications, and treatment” Radiographics, vol. 37, no. 2, pp. 628–651, 2017.

[5] S. Cho, Y. Sung, and M. Kim, “Missed rib fractures on evaluation of initial chest ct for trauma patients: pattern analysis and diagnostic value of coronal multiplanar reconstruction images with multidetector row ct” The British journal of radiology, vol. 85, no. 1018, pp. e845–e850, 2012.

[6] L. Yao, X. Guan, X. Song, Y. Tan, C. Wang, C. Jin, M. Chen, H. Wang, and M. Zhang, “Rib fracture detection system based on deep learning” Scientific reports, vol. 11, no. 1, p. 23513, 2021.

[7] M. Wu, Z. Chai, G. Qian, H. Lin, Q. Wang, L. Wang, and H. Chen, “Development and evaluation of a deep learning algorithm for rib segmentation and fracture detection from multicenter chest ct images” Radiology: Artificial Intelligence, vol. 3, p. e200248, 2021.


