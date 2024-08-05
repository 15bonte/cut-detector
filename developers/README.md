# Developers README

## How to generate ground truth for training and evaluation

## Generate division movies

First, run Cut Detector on a video with both "Save cell division movies?" and "Debug mode" checked. "Save cell division movies?" will generate a TIFF file for each cell division detected in the video. "Debug mode" will create a folder "Mitoses" in the results directory that will be used to update ground truth.

## Open video and CellCounter plugin in Fiji

Annotations are performed using Fiji and plugin CellCounter. In Fiji, open a cell division movie, just as usual. Open plugin by typing “Cell counter” in the Quick Search. Click on “Initialize” in the plugin.

## Rename categories

Create and rename categories as follows. Order is not very important, but naming is. “?” is used for categories where you are not sure of the true category.

<img src="https://github.com/15bonte/cut-detector/blob/main/developers/images/CellCounter_categories.png">

• Annotate
One annotation should correspond to one midbody. Just select the corresponding category and click on the image. A marker will appear with the class number.
If there is no mid-body, do not annotate.

It is not important on which channel you annotate, but do not make the same annotation on two different channels.
If you have made a mistake, click on “Delete” to remove the last marker.
If you want to remove a marker which is not the last one, move back to the frame, select “Delete mode” and the corresponding class, and click on the marker to remove it.

• Save
Once you have annotated everything, save your results with “Save markers”. Keep default name for the file. It should start by “CellCounter\_”.
