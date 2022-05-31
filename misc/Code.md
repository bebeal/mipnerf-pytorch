# Code Notes

## Datasets

* `NeRFDataset` is a generic PyTorch `Dataset` object that you can inherit from to implement your own dataset to train/visualize a mip-NeRF.
* The basic idea is that the dataset should output a `Ray` ($\sim$ `self.rays`) object, and if you're training the corresponding RGB pixel ($\sim$ `self.images`) label for that ray
  * `Ray` is a named tuple defined in `ray_utils.py` and consist of the following:
    * `origins`: The 3D Cartesian Coordinate indicating the starting position of the ray
    * `directions`: The 3D Cartesian Vector indicating the direction of the ray
    * `viewdirs`: The 3D Cartesian Vector indicated the viewing direction for each ray.  This is only used if NDC rays are used, as otherwise
    `directions` is equal to `viewdirs`.
    * `radii`: The radii (base radii for cones) of the rays.
    * `lossmult`: A mask used to implement the ray regularization loss (essentially a weighted MSE)
      * Only used for Multicam dataset, for other datasets the mask is all 1's which corresponds to just taking the mean
    * `near`: Near plane
    * `far`: Far plane
* This Ray object is the input to the model. When a trained model is used to render a scene, we generate these rays from arbitrary poses
* There are 3 splits you can split the dataset into
  * `"train"`: Used during training, outputs (rays, pixels) that the model uses to train
  * `"test"`: Used during evaluation/testing, outputs (rays, pixels) that the model hasn't been directly trained on
  * `"render"`: Used on a trained model to "render" a scene, generated arbitrary poses and outputs (rays) used in `visualize.py` to create a video of the scene
* The dataset essentially calls one of the following function during construction.
  * Both functions have a subsequent call to a specific poses function to generate poses, and then finally a call to `generate_rays()` to generate rays from the poses. The call to the poses function assumes it will initialize `self.h`, `self.w`, `self.focal`, `self.cam_to_world`, and `self.images` if training.
  * `generate_training_rays()` for splits `["train", "test"]`
    * `generate_training_poses`: Handles generating the training poses, typically reads files from disk and sets up corresponding instance variables; Every child object must implement this function.
  * `generate_training_rays()` for splits `["render"]`
    * `generate_render_poses`: Handles generating potentially arbitrary poses used to render scenes using a trained mip-NeRF
* `generate_training_rays()` is technically the only function you have to implement if you inherit from this object but the default implementation for the other functions might not work the way you expect for you dataset.

### Default Datasets

There are 3 default datasets

* `LLFF`: Real life image datasets, originally curated by authors from another paper [Local Light Field Fusion](https://github.com/fyusion/llff), but some additional scenes were curated from the authors of the NeRF/mip-NeRF papers. Assumes forward-facing scenes. 
* `Blender`: Synthetic datasets created from blender models curated by the authors of the NeRF/mip-NeRF papers.
* `Multicam`: A multi-scaled variant of the Blender dataset.

## Poses

* Poses are stored as $3 \times 5$ numpy arrays that are built using a $3 \times 4$ camera-to-world transformation matrix, and a $3 \times 1$ vector containing simple pinhole camera intrinsics.
  * `cam_to_world` transformation matrix: 
$$
\begin{bmatrix}
r_{0} & r_{3} & r_{6} & t_{0}\\
r_{1} & r_{4} & r_{7} & t_{1}\\
r_{2} & r_{5} & r_{8} & t_{2}
\end{bmatrix}
$$ where $\left[r_{0}, r_{1}, r_{2}\right]$ corresponds to the x-axis rotation, $\left[r_{3}, r_{4}, r_{5}\right]$ corresponds to the y-axis rotation, $\left[r_{6}, r_{7}, r_{8}\right]$ corresponds to the z-axis rotation, and $\left[t_{0}, t_{1}, t_{2}\right]$ corresponds to the translation.
  * camera intrinsics: 
$$
\begin{bmatrix}
\text{height} \\
\text{width}  \\
\text{focal}
\end{bmatrix}
$$

## Config

### Basic Hyperparameters

* `--log_dir`:
  * Default: `"log_dir"`
  * Description: Where to save model/optim weights, tensorboard, and visualization videos to
* `--dataset_name`
  * Default: `"blender"`
  * Options: `["llff", "blender", "multicam"]`
  * Description: Which dataset to use
* `--scene`
  * Default: `"lego"`
  * Description: Which scene from the dataset to use

### Model Hyperparameters

* `--use_viewdirs`
  * Default: `True`
  * Description: Use view directions as a condition
* `--randomized`
  * Default: `True`
  * Description: Use randomized stratified sampling
* `--ray_shape`
  * Default: `"cone"`
  * Options: `["cone", "cylinder"]`
  * Description: The shape of cast rays. "cylinder" used for llff dataset, "cone" used for rest
* `--white_bkgd`
  * Default: `True`
  * Description: Use white as the background, black otherwise. False used for llff, True used for rest
* `--override_defaults`
  * Default: `False`
  * Description: Override default dataset configuration. By default, if the llff dataset is chosen the config will change default configurations like `--white_bkgd` and `--ray_shape` to match the default llff dataset configuration, unless this is specified and then it will use whatever you set them to
* `--num_levels`
  * Default: `2`
  * Description: The number of sampling levels
* `--num_samples`
  * Default: `128`
  * Description: The number of samples per level
* `--hidden`
  * Default: `256`
  * Description: The width of the MLP
* `--density_noise`
  * Default: `0.0`
  * Description: Standard deviation of noise added to raw density. "1.0" used for llff datset, "0.0" used for rest
* `--density_bias`
  * Default: `-1.0`
  * Description: The shift added to raw densities pre-activation
* `--rgb_padding`
  * Default: `0.001`
  * Description: Padding added to the RGB outputs
* `--resample_padding`
  * Default: `0.01`
  * Description: Dirichlet/alpha "padding" on the histogram
* `--min_deg`
  * Default: `0`
  * Description: Min degree of positional encoding for 3D points
* `--max_deg`
  * Default: `16`
  * Description: Max degree of positional encoding for 3D points
* `--viewdirs_min_deg`
  * Default: `0`
  * Description: Min degree of positional encoding for viewdirs
* `--viewdirs_max_deg`
  * Default: `4`
  * Description: Max degree of positional encoding for viewdirs

### Loss and Optimizer Hyperparameters

* `--coarse_weight_decay`
  * Default: `0.1`
  * Description: How much to downweight the coarse loss
* `--lr_init`
  * Default: `1e-3`
  * Description: The initial learning rate
* `--lr_final`
  * Default: `5e-5`
  * Description: The final learning rate
* `--lr_delay_steps`
  * Default: `2500`
  * Description:  The number of "warmup" learning steps
* `--lr_delay_mult`
  * Default: `0.1`
  * Description: How much sever the "warmup" should be
* `--weight_decay`
  * Default: `1e-5`
  * Description: Optimizer (AdamW) weight decay

### Training Hyperparameters

* `--factor`
  * Default: `1`
  * Description: The downsample factor of images. `1` for no downsampling
* `--max_steps`
  * Default: `200_000`
  * Description: The number of optimization/backprop/training steps
* `--batch_size`
  * Default: "2048"
  * Description: The number of rays/pixels in each training batch
* `--do_eval`
  * Default: `True`
  * Description: Whether or not to make a validation/evaluation dataset and run inference every `--save_every` number of steps
* `--continue_training`
  * Default: `False`
  * Description: To continue training
* `--save_every`
  * Default: `1000`
  * Description: The number of steps to save a checkpoint, and run evaluation if `--do_eval`
* `--device`
  * Default: `"cuda"`
  * Description: Device to load model and data to. Loads a batch of data at a time. 

### Visualization Hyperparameters
* `--chunks`
  * Default: `8192`
  * Description: Size of chunks to split rays into during visualization rendering. Does not affect quality of video. Make as big as possible without OOME. `8192` $\sim$ `10`GB
* `--model_weight_path`
  * Default: `"log/model.pt"`
  * Description: Where to load the model weights from when calling `visualize.py`
* `--visualize_depth`
  * Default: `False`
  * Description: Whether to create a video rendering of the depth prediction when `visualize.py` is called
* `--visualize_normals`
  * Default: `False`
  * Description: Whether to create a video rendering of the normals prediction when `visualize.py` is called

## Helpful Resources
* [Pinhole Camera Model](https://www.scratchapixel.com/lessons/3d-basic-rendering/3d-viewing-pinhole-camera)
* [Generating Camera Rays](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/generating-camera-rays)
* [Standard Coordinate Systems](https://www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-generating-camera-rays/standard-coordinate-systems)
* [Computing the Pixel Coordinates of a 3D Point](https://www.scratchapixel.com/lessons/3d-basic-rendering/computing-pixel-coordinates-of-3d-point/mathematics-computing-2d-coordinates-of-3d-points)
* [Projection Matrix](http://www.songho.ca/opengl/gl_projectionmatrix.html)
