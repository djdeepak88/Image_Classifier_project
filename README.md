# Image_Classifier_project

Command Line parameters usage:-

Training data:-

1. python train.py data_dir_path --learning_rate 0.01 --hidden_units 512 --epochs 20 --gpu --arch vgg19

Supported Architectures:-
1. vgg19
2. vgg16
3. resnet18
4. densenet121
5. alexnet


Prediction on test dataset:-

1. python predict.py input_path checkpoint_path --category_names cat_to_name.json
