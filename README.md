# ROBUST FRACNET: LEARNING TO DETECT RIB FRACTURES WITH NOISE





## Abstract
This paper focuses on the detection of rib fractures, which is crucial for accurate diagnosis and timely intervention in cases of trauma-related injuries. As part of the RibFrac 2020 Challenge, the aim is to build upon the FracNet baseline model using various augmentation techniques, such as adding Gaussian noise, mirroring, or positive and negative sampling. The combined application of these techniques led to an improvement of 15% over the baseline in terms of the Free-Response Receiver Operating Characteristic (FROC) score. Potential future research could investigate the effectiveness of integrating diffusion-based techniques for enhanced semantic information extraction.

## Introduction
The past years have witnessed significant advancements in the field of medical imaging, with deep learning and computer vision being at the forefront of such developments [1, 2]. The ultimate goal of such technologies is to improve the diagnostic precision of medical imaging and automatize the diagnosis process. Automating this process would alleviate the arduous and time-consuming nature of this task, allowing professional radiologists to focus on other duties.

One specific application of such advancements comes in the form of rib fracture detection, a frequent injury resulting from various traumatic events such as accidents, falls, or sports-related incidents. In the case of rib fractures, the most common imaging modality is chest computed tomography (CT) [3], which offers a detailed representation of the ribs and the surrounding area. The increased level of detail in CT scans offers radiologists more information for accurate diagnoses. However, this also translates to a more time-intensive process, as each rib necessitates thorough analysis from multiple perspectives. Despite this careful approach, rib fracture detection from CT scans still has a misdiagnosis rate of between 19.2% to 26.8% [4, 5] which has serious implications for patients' well-being. Indeed, mortality rates tend to increase with the number of rib fractures [3], underscoring the need for accurate and early detection of rib fractures.

To this end, the 2020 Rib Fracture Detection and Classification Challenge (RibFrac Challenge) [6] aimed to provide a large-scale benchmark dataset for the analysis of automatic deep-learning models tasked with detecting and classifying around 5,000 rib fractures from 3D 660 CT scans. While the challenge has both segmentation and classification task tracks, the main focus of this paper is the segmentation part. Although significant advancements have been achieved in rib fracture detection and segmentation using deep learning models, challenges persist. Wu et al. [7] highlighted difficulties rooted in rib fracture data acquisition constraints, such as privacy, ethical considerations, robustness and generalizability. In this study, we seek to experiment with data augmentation and sampling methodologies in order to explore their influence on model robustness, generalisation capability and prevent model overfitting. Our contributions can be broken down into two main research questions:

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

[1] Y. Zhan, Y. Wang, W. Zhang, B. Ying, and C. Wang, “Diagnostic accuracy of the artificial intelligence methods in medical imaging for pulmonary tuberculosis: A systematic review and meta-analysis” Journal of Clinical Medicine, vol. 12, no. 1, 2023.[Online]. Available: https://www.mdpi.com/2077-0383/12/1/303

[2] R. Aggarwal, V. Sounderajah, G. Martin, D. S. Ting, A. Karthikesalingam, D. King, H. Ashrafian, and A. Darzi, “Diagnostic accuracy of deep learning in medical imaging: a systematic review and meta-analysis” NPJ digital medicine, vol. 4, no. 1, p. 65, 2021.

[3] B. S. Talbot, C. P. Gange Jr, A. Chaturvedi, N. Klionsky, S. K. Hobbs, and A. Chaturvedi, “Traumatic rib injury: patterns, imaging pitfalls, complications, and treatment” Radiographics, vol. 37, no. 2, pp. 628–651, 2017.

[4] S. Cho, Y. Sung, and M. Kim, “Missed rib fractures on evaluation of initial chest ct for trauma patients: pattern analysis and diagnostic value of coronal multiplanar reconstruction images with multidetector row ct” The British journal of radiology, vol. 85, no. 1018, pp. e845–e850, 2012.

[5] L. Yao, X. Guan, X. Song, Y. Tan, C. Wang, C. Jin, M. Chen, H. Wang, and M. Zhang, “Rib fracture detection system based on deep learning” Scientific reports, vol. 11, no. 1, p. 23513, 2021.

[6] L. Jin, J. Yang, K. Kuang, B. Ni, Y. Gao, Y. Sun, P. Gao, W. Ma, M. Tan, H. Kang, J. Chen, and M. Li, “Deep-learning-assisted detection and segmentation of rib fractures from ct scans: Development and validation of fracnet” EBioMedicine, 2020.

[7] M. Wu, Z. Chai, G. Qian, H. Lin, Q. Wang, L. Wang, and H. Chen, “Development and evaluation of a deep learning algorithm for rib segmentation and fracture detection from multicenter chest ct images” Radiology: Artificial Intelligence, vol. 3, p. e200248, 2021.

[8] T. Falk, D. Mai, R. Bensch, &Ouml;. &Ccedil;i&ccedil;ek, A. Abdulkadir, Y. Marrakchi, A. B&ouml;hm, J. Deubner, Z. J&auml;ckel, K. Seiwald et al., “U-net: deep learning for cell counting, detection, and morphometry” Nature methods, vol. 16, no. 1, pp. 67–70, 2019


