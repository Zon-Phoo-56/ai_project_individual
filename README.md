# AI Project Folder Structure

sever_ai/
├── __pycache__/
├── data/
│   ├── CSVs/
│   ├── images/
│   ├── labels/
│   ├── data_main.py
│   └── data_preparation.py
├── sessions/
│   ├── best_model.pth
│   └── learning_curve.png
├── test_images/
├── test_results/
├── venv/
├── args.py
├── augmentations.py
├── dataset.py
├── evaluate.py
├── gpu_test.py
├── main.py
├── model.py
├── trainer.py
└── utils.py
#### Short Explanation of Each File and Folder
data/
  - data/  Contains the dataset and data processing scripts.
  - data/CSVs/  Stores CSV files used for annotations, metadata, or dataset records.
  - data/images/ Contains the training or original image files.
  - data/labels/ Stores label files for the images.
  - data/data_main.p Main script for handling or organizing the dataset workflow.
  - data/data_preparation.py Split all the data for training and validation.
  - sessions/ Contains saved training outputs and model results.
Sessions/
  -  sessions/best_model.pth Saved best trained model weights.
  -  sessions/learning_curve.png Image showing training and validation performance over epochs.

test_images/
  -  Contains images used for testing the trained model.

test_results/ 
  -  Stores prediction outputs or evaluation results from testing.
venv/
  -  Python virtual environment folder with installed dependencies.
- args.py Defines configuration settings and command-line arguments.
- augmentations.py Contains image augmentation methods for training.
- dataset.py Defines how the dataset is loaded and processed.
- evaluate.py -Runs model evaluation on validation or test data.
- gpu_test.py- Checks whether GPU is available for cuba or not.
- main.py Main entry point for running the project.
- model.py Defines the deep learning model architecture.
- trainer.py Handles the training process of the model.
- trainer.py Handles the training process of the model.
- utils.py Contains helper functions used across the project.






