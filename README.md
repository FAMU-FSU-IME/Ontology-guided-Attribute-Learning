
# Overview
## Project Title: Ontology-guided Attribute Learning to Accelerate Defect Identification for Developing New Printing Processes
## Abstract:
Identifying printing defects is vital for process certification, especially with evolving printing technologies. However, this task proves challenging, especially for micro-level defects necessitating microscopy, which presents a scalability barrier for manufacturing. To address this challenge, we propose an attribute learning methodology inspired by human learning, which identifies shared attributes among seen and unseen objects. First, it extracts defect class embeddings from an engineering-guided defect ontology. Then, attribute learning identifies the combination of attributes for defect estimation. This approach enables it to recognize previously unseen defects by identifying shared attributes, even those not included in the training dataset. The research formulates a joint optimization problem for learning and fine-tuning class embedding and ontology and solves it by integrating natural language processing, metaheuristics for exploration and exploitation, and stochastic gradient descent. In a case study involving a direct-ink-writing process for creating nanocomposites, this methodology was used to learn new defects not found in the training data using the optimized ontology. Compared to traditional zero-shot learning, this ontology-based approach significantly improves class embedding, outperforming transfer learning in one-shot and two-shot learning scenarios. This research represents an early effort to learn new defect concepts, potentially reducing the need for extensive measurements in defect identification.
### Requirements
To replicate the results and conduct further research in this project, ensure you have the following dependencies installed:
- Python 3.x
- NumPy
- SciPy
- scikit-learn
- Flair
- spaCy
- networkx
- Matplotlib
- tqdm

These requirements will provide you with the necessary tools and libraries to execute the code, perform experiments, and reproduce the results outlined in the project.
## 3.3.2: Application of the Ontology for Zero-shot Defect Classification

This section outlines the successful application of ontology-guided attribute classifiers for zero-shot defect classification. The results demonstrate the algorithm's efficacy in identifying defects in unseen classes. The experiments encompass various scenarios, with different splits between seen and unseen classes, showcasing the algorithm's performance.
This research represents a significant advancement in defect identification, especially for micro-level defects, by leveraging an attribute learning methodology inspired by human learning. It offers a scalable solution for evolving printing technologies

### Steps to Reproduce Results
To replicate the results in Table 2 for Zero-shot Defect Classification:
1. Download the "3.3.2 Table 2 Zero-shot defect classification.zip" file.
2. Extract its contents, including `res101.mat`, `att_splits.mat`, `wiki_sentences_v11.csv`, and `ZSL_using_Ontology_and_EA.ipynb`.
3. Create a "best" directory to store the current best attribute file during experiments.
4. Execute the code within `ZSL_using_Ontology_and_EA.ipynb`, following provided instructions and settings, to reproduce the results. 

## 3.3.2 Figure 9 and Figure 20 in S9: Confusion matrix for three unseen classes
Figure 9 presents a confusion matrix for true and predicted labels when three unseen classes are present in the testing data, while Supplementary S9, Figure 20 illustrates the scenario where seen and unseen classes are mixed during testing. In both cases, accuracy surpasses 0.72. The three-unseen-class zero-shot learning (ZSL) task is less challenging as it involves only three unseen classes, whereas the mixed-class scenario includes all ten classes in the testing dataset, making classification more complex. Although the mixed-class scenario yields lower accuracy, it's important to note that this study does not aim to compare results with seen-only classes. These results showcase the potential of ontology-guided attribute learning in accelerating process certification, even with minimal measurements from a new process, addressing challenges beyond conventional classification and transfer learning approaches.
### Steps to Reproduce Results:
1. Download the file "3.3.2 Figure 9 and S9 Figure 20 Confusion matrix for three unseen classes.zip."
2. Extract the contents of the zip file to a directory of your choice.
3. Inside the extracted directory, you will find the following files:
   - `res101.mat`: Data file containing features.
   - `att_splits.mat`: File containing class splits.
   - `wiki_sentences_v11.csv`: Ontology sentences.
   - `ZSL_using_Ontology_and_EA.ipynb`: Code notebook.
4. Create a directory named "best" in the same location to store the current best attribute file.
5. Execute the Jupyter notebook "ZSL_using_Ontology_and_EA.ipynb" to reproduce the results.

## 3.3.3 Effect of ontology exploration rate
The ontology exploration rate 0 < Th0 < 1 in the two-stage optimization plays a crucial role in determining the ontology's likelihood of being re-structured to incorporate new information. This exploration rate influences how the ontology adapts to new information, and this section explores its impact on Zero-Shot Learning (ZSL) accuracy. All other parameters, including epochs and termination criteria for iterations, remain constant at their previously optimized levels.
A higher exploration rate implies more frequent ontology re-structuring, leading to significantly different embeddings and a broader search for optimal solutions. Conversely, a lower exploration rate places more emphasis on fine-tuning class embeddings and the initial ontology. Figure 10 displays the 95% confidence interval (CI) of the average accuracy estimated from multiple samples for each Th0.  The results indicate that lower ontology exploration rates (0.1-0.4) yield lower average accuracy.
### Steps to Reproduce Results:
1. Download the file "3.3.3 Figure 10 Effect of ontology exploration rate.zip."
2. Extract the contents of the zip file to a directory of your choice.
3. Inside the extracted directory, you will find the following files:
   - `res101.mat`: Data file containing features.
   - `att_splits.mat`: File containing class splits.
   - `wiki_sentences_v11.csv`: Ontology sentences.
   - `ZSL_using_Ontology_and_EA.ipynb`: Code notebook.
4. Create a directory named "best" in the same location to store the current best attribute file.
5. Open the Jupyter notebook "ZSL_using_Ontology_and_EA.ipynb."
6. In the notebook, find and modify the value of Th0 to be 0.1 – 0.9 and repeat each experiment 10 times.
7. Execute the modified notebook to reproduce the results for the effect of ontology exploration rate.

## 3.3.4 Robustness against data noises
A robustness study was conducted to assess the methodology's performance in the presence of various levels of noise added to existing images. Gaussian white noise with a variance ranging up to 1.2 was introduced to each image. Figure 11  (lower panels) provides a visual example of a noise-free reference image compared to four levels of added noise. The introduced noise has the potential to diminish shared features among classes and images while reducing visibility by blurring fine details. The noisy dataset consists of seven training classes and three unseen testing classes, with each run repeated ten times. The boxplot in Figure 11 presents the interquartile range (IQR) and median of classifier accuracy at each noise level, both for testing only unseen classes (upper left panel) and a combination of seen and unseen classes (upper right panel). Generally, the methodology's accuracy decreases as images become progressively blurred. Notably, the algorithm's performance declines rapidly when classifying mixed classes of seen and unseen during training, mainly due to bias toward the seen classes. This effect becomes more pronounced as the noise level increases, with a consistent decrease in accuracy for the classification of ten seen-unseen classes and larger variability for the classification of three unseen classes.


### Steps to Reproduce Results:
1. Download the file "3.3.4 Figure 11 Robustness against data noises.zip."
2. Extract the contents of the zip file to a directory of your choice.
3. Inside the extracted directory, you will find subdirectories with different levels of added noise, including "noise_free," "variance_0.4," "variance_0.80," and "variance_1.2."
4. Enter any of these subdirectories, e.g., "noise_free."
5. Inside each subdirectory, you will find the following files:
   - `res101.mat`: Data file containing features.
   - `att_splits.mat`: File containing class splits.
   - `wiki_sentences_v11.csv`: Ontology sentences.
   - `ZSL_using_Ontology_and_EA.ipynb`: Code notebook.
6. Create a directory named "best" in the same location to store the current best attribute file.
7. Open the Jupyter notebook "ZSL_using_Ontology_and_EA.ipynb."
8. Execute the notebook for each noise level (e.g., "noise_free," "variance_0.4," etc.) to reproduce the results for the robustness against data noises.

## 3.3.5 Comparative study against transfer learning
The proposed methodology was subjected to a comparative study against three benchmark methods within a transfer learning framework. It's essential to note that traditional transfer learning isn't designed for zero-shot learning (ZSL). Therefore, this comparison was conducted in the context of one-shot and two-shot learning, where classes had only one or two samples observed in the training data. Three popular pre-trained CNN models for image classification, namely AlexNet, VGG, and ResNet50, were utilized in this study. Supplementary S6 provides details on the transfer learning networks. To implement transfer learning, a pre-trained CNN network was obtained from an open-source repository. The last fully-connected and softmax layers of these networks were replaced with a new custom fully connected layer containing ten output neurons. During training, the pre-trained weights remained frozen and were not updated, while the added final layer was trained using the AM defect data. The proposed method was compared with transfer learning techniques, including AlexNet, ResNet50, and VGG, in one-shot and two-shot learning scenarios, where only one or two samples of a class were available in the training data. The box plots of accuracy for one-shot learning tests are presented in the left panel of Figure 12 and the results of two-shot learning experiments are reported in the right panel. In both cases, the proposed method demonstrated statistically superior performance compared to the other transfer learning methods, with ResNet50 being the least effective.

### Steps to Reproduce Results:
1. Download the file "3.3.5 Figure 12 Comparative study against transfer learning.zip" from here (https://drive.google.com/file/d/1dipaivoYE5WpDYgYLiBCl3wtN_O_WeGM/view?usp=sharing)
2. Extract the contents of the zip file to a directory of your choice.
3. Inside the extracted directory, you will find a subdirectory named "one shot learning." and "two shot learning"
4. Additionally, within the extracted directory, you will find subdirectories named "AlexNet," "VGG," "ResNet, "attribute learning""
9. Enter any of these subdirectories (e.g., "AlexNet").
5. Inside each subdirectory, you will find data files for all ten defects and split files "train.tex" and "test.tex."   7. Enter the "attribute learning" subdirectory.
6.  Inside this subdirectory, you will find the following files:
   - `res101.mat`: Data file containing features.
   - `att_splits.mat`: File containing class splits.
   - `wiki_sentences_v11.csv`: Ontology sentences.
   - `ZSL_using_Ontology_and_EA.ipynb`: Code notebook.
   - Open the Jupyter notebook "ZSL_using_Ontology_and_EA.ipynb."
7. Create a directory named "best" in the same location to store the current best attribute file.
8. Execute the notebook for the "attribute learning" section and for each transfer learning method (e.g., AlexNet, VGG, ResNet) to reproduce the results for the comparative study against transfer learning.


