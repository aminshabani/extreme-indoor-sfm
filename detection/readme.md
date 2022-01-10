# Detection Module

You can download the checkpoints from [Google Drive](https://drive.google.com/file/d/172E2vJ4x_wqH6OYNLYI_l6MH-bYbOnX6/view?usp=sharing).
### Generating the detection results
To generate the predictions:
```
python -m detection.test
```
The above command create a `detection_preds` folder for each house in the dataset. You can also pass `det_save_image` to also save the predictions as images in the same directory.
```
python -m detection.test --det_save_images 
```

### Evaluation
To evaluate the performance:
```
python -m detection.test --det_is_eval
```

### Training on new dataset
For the training on your data, please see the [panorama.py](panotools/panorama.py).


