# pytorch-combinators
Combinators are a concept from functional programming. 
The term describes a form of higher order function which takes one or more functions as parameters (possibly in addition to other data) and gives back a new function. 
They have proved to be very effective at building modular, composable toolboxes for libraries and frameworks.

The idea that drives this little project is that neural networks can be thought of as *learning functions*. 
If we take a PyTorch module, we usually think of it as a class, but that is to manage data, the learning parameters etc... inside it that will be trained. 
The execution is the forward method, and this is usually a rather pure transformation, once training has taken place. 
So, the proposal is, that we can build a small toolbox of combinators that abstract common patterns that are useful in building up bigger neural networks. 

ADVANTAGES OF CONCEPT? 
     PARAMETERS CHANGE BEHAVIOR   / PARACHUTE IN COMPLEX SUB PROCESSES, INCLUDING LEARNING PROCESSES
     STRUCTURE OPTIONS

# Examples

## ResNet34
DIAGRAM + CODE

From the paper by [He et al (2016)](#resnet_reference) 

<p align="center">
<img src="./images/Resnet34_layers.png" alt="Bootstrap logo" width="70%" height="70%"/>
</p>



# References

<a id="resnet_reference"></a>
- He, K., Zhang, X., Ren, S., Sun, J., 2016. Deep residual learning for image
recognition, in: 2016 IEEE Conference on Computer Vision and Pattern
Recognition (CVPR), pp. 770–778. [doi:10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).




