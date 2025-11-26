# BorderAttack

"Adversarial Attacks and Defenses for Image Border Perturbations"
<br>
*Gergő Török, János Hadabás and István Megyeri*
<br> <br>
In this repository we present the code for our BorderAttack adversarial attack. 

## Dependencies

The repository is written using python 3.8. The dependecies can be installed using the following command:
<br> <br>
```pip install -r requirements.txt```

## Evaluation

The two CIFAR-10 checkpoints presented in the paper are uploaded to [BorderAttack](https://drive.google.com/drive/folders/19cKOzkn_vlK_vhUBDreLBKf_DQVFsQt2?usp=sharing). These can be evaluated on the CIFAR-10 validation set using the following command:
<br> <br>
```python eval-ba.py --data-dir data_dir --log-dir log_dir --desc <model_name>```

## Setting the Parameters

### Losses
Currently 4 different losses are implemented in BorderAttack: Cross Entropy, Carlini-Wagner, Margin and Difference Loss Ratio. 
<br>
A loss combination should be given as a list of losses eg. ```['CE', 'CW']```. We define two separate loss combinations for untargeted and targeted evaluations, use the ```losses``` and the ```losses_t``` parameters accordingly. 

### Border Size
The border size used by the attack can be set using the ```bs``` parameter. By default it is set to 1.



