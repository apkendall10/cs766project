# Convolutional Neural Networks in Gaussian Blur Deconvolution
Amos Kendall and Juan Rios

## Problem Statement

Image blur is a pervasive problem in image capture. Noise from blurry images decreases the visual appeal of pictures, and makes computer vision algorithms less reliable. The challenge of blur recovery is a classic ill-posed problem, often modeled as

<p align="center">
B = K*I + N
</p>

Where B is the known blurred image, K is the blur kernel, I is the unknown true image, and N is unknown noise. The challenge is that a method must predict K, I, and N from a given example or set of examples [1].

Fundamentally, there are two different types of blur problems - blind blur and non-blind blur. Blind blur is where the blur kernel, K, is not known a priori. Non-blind blur occurs when the blur kernel is known [2]. Blind blur problems are both more challenging and more common, and will be the focus of this project.

## Background
Blur recovery is important because it can be used with popular consumer products like cameras or phones and can aid other tech and software initiatives. Additionally, deblurring is important in scientific applications. For example, in Astronomy image data of star systems is subject to blurring effects. Undoing these effects is important in creating more accurate results [3]. Another example is the use of deblurring in medical imaging such as X-ray, mammographies, and ultrasound images for more accurate medical diagnosis [4].

The classical approach to blind blur problems is to look at a single image and use probability based estimates to predict the true image. Typical methods include Bayesian Inference Frameworks, which find an optimal solution by maximizing a hypothesis based on given evidence. Another popular approach is to use regularization techniques such as Tikhonov-Miller regularization, which attempt to convert ill-posed problems to well-posed problems by constraining the solution space. Other methods include Homography-Based methods, Sparse Representation-Based methods, and Region-Based methods [1].

With advances in computing power and refinement of neural network theory, especially deep, convolutional neural networks (CNN), there have been various attempts to use a collection of blurred images to learn more details of the problem. The key insight is that if some features of image blur are shared between different images, analyzing multiple images gives more information when trying to discover the unknowns, K, I and N. Initial approaches aimed to train a network to learn the blur kernel, K, from synthetic datasets where K would be known, but varied, to allow for a generalized learning of K. Examples include Schuler [5], who iteratively uses CNN to extract features and then estimate the kernel, and Sun [6] who uses a CNN to estimate motion blur vectors.

Recently, there has been more focus on end-to-end networks that take the blurred image, B, as input and attempt to directly produce the true image, I. The advantage of these end-to-end networks is that the learning considers both the kernel estimation on the image recovery in a continuous form, and they do not rely on a known blur kernel. These models can gracefully handle more complicated, spatially varying kernels, and tend to be more robust to noise and saturation [7]. State of the art learning methods use CNNs with additional architecture designed to maximize the learning of shared information between images.  Nah et al train multiple CNNs using different image scales, where a smaller scale network output is also used as input for the next larger scale. Each network is then updated via back propagation. The effect is to learn abstract features of the image while preserving localized information needed to reconstitute the pixels of the true image [7]. Tao et al use a similar method of combining networks of multiple scales, but with added connected recurring long short term memory units in the middle of each network to add more connections between the scales. They also applied a form of encoding and decoding within a single scale to further magnify the learning of both abstract and local features [8]. Zhu et al use generative adversarial networks (GAN) to learn the loss function while also training a network to generate non-blurry images. Their generator network is based on a UNET framework which uses encoding and decoding with layer skipping to combine local and abstract features. The discriminator then learns to examine a collection of local patches to determine if the generated image comes from the ground truth sample or from a generated de-blurred image. The ability to learn the loss function in this way allows for more powerful optimization of the deblurring problem [9].

## Our Approach 

Our goal was to investigate how learning based approaches could be integrated with classical non-learning approaches.

### Phase I

Initially, we focused on a simple setting -  gaussian blur applied uniformly to the entire image. For our initial dataset, we used 25000 images from imagenet [10]. For testing and training, we applied a blur to these images. The gaussian blur standard deviation (sigma) was drawn randomly from (the absolute value of) a normal distribution with mean 0 and sdv 5. 

To determine a baseline for performance, the Maximum Likelihood algorithm is used for blind deconvolution, and the Richardson-Lucy (RL) algorithm was chosen for non-blind deconvolution. These are standard methods for non-blind and blind deblurring. Both algorithms are based on the Bayesian Inference Frameworks [1].

We also developed two different neural network architectures. The first one takes a blurry image and predicts the blur kernel. Then we use the blur kernel to apply the non-blind RL algorithm. Refer to this method as CNN + RL.

To predict the gaussian blur kernel, we trained a CNN with 3 convolutional layers and 3 fully connected layers. The CNN was given a blurry image as input with a gaussian kernel size as a label.

The second network is an end-to-end network that takes in the blurry image and produces a prediction of the true underlying image. This network is based on UNETs [11] commonly used for pixel segmentation. UNETs are a powerful method to learn a combination of abstract features and local details. They use convolutional layers (often called encoding) to build increasingly abstract feature maps, and then combine upsampling with the already generated less abstract feature maps to decode the learned features into an image. Our UNET architecture used 3 encoding and 3 decoding layers.

### Phase I Results

We compare the performance of our methods using the mean PSNR and SSIM as our evaluation metrics. For each dataset, we used a test set that was unseen by the various learning algorithms.

The results compare our non-learning baseline, with the CNN blur kernel estimator + RL, and the End to End Unet based approach. Note that blind deblur takes as input a guessed filter based on 1 sigma, which gives the best blind deblur performance based on our blur kernel distribution. Additionally, the blind deblur method will output a recovered kernel.

| Method | PSNR | SSIM |Runtime on testset (s) |
|------|-----|-----| ----- |
|Baseline blind deblur |23.15 | 0.7541| 182.44 |
|CNN + RL |24.18 |0.7669 | 72.19 |
|End to End UNET | 21.58| 0.7526| 781.67 |

The CNN + RL achieved the best results for both SSIM and PSNR in the least amount of time. A 2-tailed t-test resulted in PSNR differences as statistically significant, while SSIM differences as not statistically significant. Nevertheless, the CNN + RL remains the best method for this setting. Additionally, the CNN is substantially better at recovering a kernel than blind deblur. We compare a kernel distance MSE of 0.02 for CNN + RL, to a MSE of 0.07 for blind deblur. 

The performance of the end-to-end network was in line with both blind and CNN + RL methods. Subjectively, we believe it recovers an image that looks less blurry to the human eye especially for large sigma values. This is consistent with some criticism of PSNR as an accurate measure of deblur quality [13].

<img src= "PhaseI_figure1.PNG">
Figure 1 The results of the three deblurring methods performed on 4 images.

### Phase II 

The second phase addresses the issue of spatially varying blur. In natural settings, blurs are rarely applied uniformly to an image. Instead the blur is typically local. For example, blur caused by varying depth of field will only be present in the parts of the scene that differ in depth from the plane of focus. 

We used the true vs. blur dataset generated by Nah[7]. This dataset consists of pairs of blurry and sharp images. The blurry images have spatially varying blur. It is important to note that unlike in phase I, we do not have a known blur kernel, and it is not limited to gaussian blur. Also, to improve training performance, we shrunk each image by a factor of 5.

Our first approach assumes that for a small image patch, the blur is constant. Given this, the network in the CNN + RL was retrained to predict the kernel of small image patches. We partition each large blurry image into 25 (5x5) segments, and predict the kernel for each segment. Each segment is then recovered through the RL algorithm with its appropriate prediction as input, and the large image is rebuilt from stitching the individually-recovered patches.

For the end to end network, we were able to use the same architecture as in phase I. However we extended our results by implementing a GAN approach where the generator network was given a blurry image and tried to construct a denoise image. The discriminator network was fed both true sharp images and denoised images from the generator, and tried to distinguish between the two sets. The descriminator’s success was then used as the basis of the loss for both networks. The architecture for the generator was similar to that of the UNET we used, except with an additional skip layer connection between the blurry input image and the final upsampled layer. This idea was motivated by Resnet architecture, which suggests that learning the difference between the input and output can be easier than learning a function that maps the input to the output [12]. For the discriminator, we used 6 convolutional layers with leaky ReLU and batch normalization.

### Phase II Results

Results show the metrics of each approach compared to the sharp image. As a baseline, we compare the blurry images to the sharp images. One surprising result was the high score for PSNR and SSIM for the blurry baseline. This is due to a pair of factors. First, since the blur was captured in a realistic way that was only present in small areas of the image, most of the blurry image was a perfect match with the sharp image. Second, since we resized the images both the sharp and blurry images result from an interpolation of neighboring pixels in the larger image, which reduces the difference between the two images. A 2-tailed t-test reveals the metrics difference for all methods are statistically significant unlike phase I. This is a result of a larger evaluation set size and smaller standard deviations.

| Method | PNSR | SSIM |Runtime on testset (s) |
|------|-----|-----| ----- |
| Blurry Baseline    | 31.9200  |0.9574    | 0 |
| CNN + RL + Stitching    |  | | |
| UNET | 29.0705   | 0.9531 |2802.63 |
| GAN |24.7497 |0.8800 |947.4906 |

The UNET performed the best overall in regards to SSIM, but lagged behind in PSNR. By visual observation the quality of the UNET is very good. The consistent performance of the UNET across phases highlights the flexibility of the end-to-end approach. We believe that this network architecture could be used successfully in any deblur learning setting.

<img src= "PhaseII_figure2.PNG">
Figure 2 The results of the three deblurring methods performed on 4 images.


The GAN underperformed compared to the traditional UNET structure, but it did provide two interesting results. First, the performance continued to improve with additional training, suggesting that more training time could help close the gap. The table below shows the performance based on the number of training epochs, where each Epoch cycled through all 24000 training images.

| Training Epochs | SSIM | PNSR |
|------|-----|-----|
| 2    | 0.7800  |21.7142    |
| 3    | 0.8011  | 22.8159  |
| 4    | 0.8800   | 24.7497  |

Second, since the GAN generator is trained specifically to fool the GAN discriminator, it tended to produce image artifacts that fooled the discriminator, but did not actually improve the deblur process. We believe that additional training would help the discriminator discriminate between these artifacts and true sharp images.

The performance of CNN + RL was comparable to the UNET with better PSNR and worse SSIM. This result was surprising because the CNN was trained to predict uniform gaussian blur, and the images had spatially varying and diverse types of blur. These results show that the assumption of localized blur is reasonable, and that various blur types can be modeled as gaussian.

The runtime was significantly higher than other methods for two reasons. First, the algorithm was applied to a full sized image instead of the smaller images used for the UNET and GAN. Second, for each image, the CNN had to predict 25 kernels, and the RL algorithm recovered 25 patches, multiplying the computational time. 

Another difficulty arose from the RL algorithm when stitching the images together. The RL algorithm will produce a recovered image with very distinct artifacts that are present on the image edges. When the large image was reconstructed, the grid-like structure from these artifacts was clearly visible. To improve this, the image patch that is passed to the RL algorithm is slightly larger than the original patch predicted on. The resulting image patch is then cropped from this slightly larger patch, and used to reconstruct the larger image. This method provided a marginal increase in metrics, a substantial improvement in subjective visual quality, but at the cost of increased computational time.


## Challenges

One challenge in tuning the hyperparameters for the end-to-end networks is that the training time on our current hardware is costly. It took 16 hours to train the end-to-end network over 2 iterations of 25,000 relatively small images. This time horizon increases the cost of experimenting with different hyperparameters, training image sets. and network architecture. This was especially clear with the GAN network, where we did not achieve convergence in performance.

## Conclusion

Phase I shows the CNN + RL performs strongly in predicting the blur sigma for an image patch and is the best method for that setting. Phase II metrics for the recovered image were worse than the blurry baseline. This approach is limited to the deblurring capabilities of the RL algorithm and the scope of CNN training. For example, it requires prior knowledge of the kernel sigma. Thus the training set is limited to artificial blurs. Visually, the recovered images look sharper. But some of the artifacts left by the RL algorithm and the stitching method remain. Future work could include producing and training with natural datasets with varying blurs and known kernels, and exploring different metrics.

Our results also show that end-to-end methods can outperform non-learning based approaches and are likely to outperform blur prediction combined with other algorithms, especially in realistic situations where the blur kernel is not known. Further improvements can be made by longer training and more hyperparameter tuning, especially for the GAN network where the performance depends on both the generator and discriminator network performance.

One downside to end-to-end and learning based approaches is the amount of data required and the amount of training time needed. Even the computational time of evaluation can be significant in end-to-end methods. In narrow situations like uniform blur, there is space for faster algorithmic based methods.


## References

[1] https://arxiv.org/pdf/1409.6838.pdf

[2] https://arxiv.org/ftp/arxiv/papers/1710/1710.00620.pdf

[3] https://svs.gsfc.nasa.gov/2796

[4] https://pdfs.semanticscholar.org/6121/aa87089eee8d85109b5a291cd1b39ebd2639.pdf

[5] https://arxiv.org/pdf/1406.7444.pdf

[6] https://arxiv.org/abs/1503.00593

[7] https://arxiv.org/abs/1612.02177

[8] https://arxiv.org/abs/1802.01770

[9] https://arxiv.org/abs/1611.07004

[10] http://www.image-net.org/

[11] https://arxiv.org/abs/1505.04597

[12] https://arxiv.org/abs/1512.03385

[13] https://www.mathworks.com/help/images/image-quality-metrics.html

[14] https://fled.github.io/paper/blur.pdf
