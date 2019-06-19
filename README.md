# object-detection
<p>Step 1 - Download COCO validation images, 2017, (1GB) from their site here: http://cocodataset.org/#download </p>
<p>Step 2 - Download COCO 2017 Stuff Train/Val annotations [1.1GB] also from their site here: http://cocodataset.org/#download </p>
<p>Step 3 - Run "python3 generate_data.py" to generate csv, hdf5 files</p>
<p>Step 4 - Run "python3 compile.py" to run the object detection model and see that the rpn part is broken</p>

<p>Current issues: The resnet feature detection part works great but the region proposal network is broken, I can't figure 
out how to backpropagate the loss function through the "mini-batch" of anchors as described in the faster r-cnn paper. I may come back 
to this project later, if anyone wants to work on it then please go ahead.</p>
<br>
<p>Current setup: python3, tensorflow version 2.0, I havn't tested it on gpu yet, only cpu.</p>

