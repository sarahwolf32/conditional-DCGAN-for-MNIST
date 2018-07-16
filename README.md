# Conditional DCGAN for MNIST

This is a generative model for the hand-written digits of the MNIST dataset. It combines the DCGAN architecture recommended by [Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks (Radford et al)](https://arxiv.org/pdf/1511.06434.pdf) with the inputting of labels suggested in [Conditional Generative Adversarial Nets (Mirza)](https://arxiv.org/pdf/1411.1784.pdf).

## Why a Conditional GAN?

In my [last project](https://github.com/sarahwolf32/DCGAN-for-MNIST), I used a DCGAN to generate MNIST digits in an unsupervised fashion - although MNIST is a labeled dataset, I threw away the labels at the beginning and did not use them. This worked, but of course those labels held a great deal of useful information. It would have been nice to allow the GAN to benefit from that additional input, and it would have also been nice to be able to specify which digit I wanted the trained generator to create. 

Conditional GANs tackle both of these shortcomings by feeding the labels into both the Generator and Discriminator. 

[insert diagram here]

This has a couple of effects. For example, in the unsupervised DCGAN, the random vector <i>z</i> input controlled everything about the resulting digit - including which digit it was. Since that role is taken over by the labels in a conditional GAN, the <i>z</i> input here encodes all the <i>other</i> features (rotation, style, and so on). 

Feeding in the labels also affected training. I found that the architecture that had worked in my last project quickly suffered from mode collapse when I used the corresponding version here. Apparently, the labels made it easier for the Discriminator to do its job, allowing the Discriminator to "win" the minimax game prematurely. The generator lost the gradients it needed to learn and started outputting identical black images. 

Using fewer layers and larger filters stabilized training. See ```trainer/architecture.py``` for details.

## Results

## Acknowledgements




