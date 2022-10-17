# Temporal Graph-Based CNNs (TG-CNNs) for Online Course Dropout Prediction

__Authors__: Zoe Hancox and Samuel D Relton

![Link to the paper](https://link.springer.com/chapter/10.1007/978-3-031-16564-1_34)

## What is in this repository?

This repository contains the code used in the paper __Temporal Graph-Based CNNs (TG-CNNs) for Online Course Dropout Prediction__.


Architecture of the unbranched TG-CNN model used in this code:

![TG-CNN architecture](images/model_architecture_diagram.png)

## How do I run these scripts?

You can run these models by installing the requirements 

```
pip install -r requirements.txt
```
and following the steps below...

The Jupyter Notebook `dataframe_save.ipynb` file converts the `mooc_actions.tsv` MOOC data file into a dataframe `.pkl` file (either containg the last 100 actions `targetid_and_scaled_time_last100.pkl`, or all 505 actions `targetid_and_scaled_time_all.pkl`) with columns user ID, action ID and the time between the actions. These processing scripts are not required to run the model, but can be used to change the number of actions if desired. The `labels_save.ipynb` file converts the final dropout label from `mooc_action_labels.tsv` into the `labels.npy` file for easier input when training the model.

To train and test the proposed unbranched TG-CNN model (with &gamma;=1) run

```
python models\mooc_without_variable_gamma.py
```

To train and test the proposed unbranched TG-CNN model (with a variable &gamma; parameter) run

```
python models\mooc_with_gamma.py
```

To train and test the proposed branched TG-CNN model (with &gamma;=1) run

```
python models\branch_mooc_without_variable_gamma.py
```

To train and test the proposed branched TG-CNN model (with a variable &gamma; parameter) run

```
python models\branch_mooc_with_gamma.py
```


The TG-CNN models could not be cross-validated due to the computation requirements exceeding the time limit of the high performance computer used in this project. Future work will involve validation of this model in a more robust manner.

To train the baseline LSTM model with 5-fold cross-validation run

```
python models\baseline_LSTM.py
```

To train the baseline RNN model with 5-fold cross-validation run

```
python models\baseline_RNN.py
```

Torch version 1.7.0, tensorflow 2.4.1, numpy 1.19.2, pandas 1.2.4, scikit-learn 0.23.1 and cuda 10.2.89 were used for creating these algorithms (as given in the `requirements.txt` file). This work made use of the facilities of the N8 Centre of Excellence in Computationally Intensive Research (N8 CIR) provided and funded by the N8 research partnership and EPSRC (Grant No. EP/T022167/1). The Centre is co-ordinated by the Universities of Durham, Manchester and York. A V100 GPU was used with an IBM POWER9 processor.

## Graph Network and Model Architecture Visualisation 

Graph network of actions and the elapsed time between them put into a 3D tensor form example:

![graph network, coordinates and 3D tensor appearance](images/tensor_building_2.png)



## References

Hancox, Z., Relton, S.D. (2022). Temporal Graph-Based CNNs (TG-CNNs) for Online Course Dropout Prediction. In: Ceci, M., Flesca, S., Masciari, E., Manco, G., Raś, Z.W. (eds) Foundations of Intelligent Systems. ISMIS 2022. Lecture Notes in Computer Science(), vol 13515. Springer, Cham. https://doi.org/10.1007/978-3-031-16564-1_34

If you make use of this code, the TG-CNN algorithm or the paper please cite the following paper:
```
@InProceedings{10.1007/978-3-031-16564-1_34,
author="Hancox, Zoe
and Relton, Samuel D.",
editor="Ceci, Michelangelo
and Flesca, Sergio
and Masciari, Elio
and Manco, Giuseppe
and Ra{\'{s}}, Zbigniew W.",
title="Temporal Graph-Based CNNs (TG-CNNs) for Online Course Dropout Prediction",
booktitle="Foundations of Intelligent Systems",
year="2022",
publisher="Springer International Publishing",
address="Cham",
pages="357--367"
}
```

### Useful Links to the Original ACT MOOC Dataset and Corresponding Papers
* [Kumar et al's JODIE model](https://snap.stanford.edu/jodie/)
* [ACT MOOC Dataset](https://snap.stanford.edu/data/act-mooc.html)


