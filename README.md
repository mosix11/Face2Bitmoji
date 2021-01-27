# Face2Bitmoji
In this project, I'm going to use state-of-the-art approaches to Unpaired Image-to-Image Translation to tackle the problem of transforming human faces to cartoon faces (Bitmoji Faces) and try to improve them if possible.

## Database
Due to the lack of an organised dataset for Bitmoji faces, I made my own [Dataset](https://www.kaggle.com/mostafamozafari/bitmoji-faces?select=BitmojiDataset). It consists of 4084 Bitmojies. For human faces, I used [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ) and extracted full-frontal faces by the help of pose estimation model [FSA-NET](https://github.com/shamangary/FSA-Net).

## Current state-of-the-art approaches to this problem
### 1) LAcycleGAN
This approach is based on [CycleGAN](https://github.com/junyanz/CycleGAN) and the authors have tried to improve CycleGAN by adding some parts to the model to improve the Geometric Transformations. The authors of the original paper have not published the codes for the model, so I tried to implement and slightly change them. There are some deficiencies in the current code but the results that it delivers are approximately the same as those of the original paper. In my experience, the suggested model is very hard to train and isn't stable enough to be considered as a good approach to this problem.
  
### 2) Contrastive Unpaired Translation (CUT)
A faster algorithm for Unpaired Image-to-Image Translation from the authors of [CycleGAN](https://github.com/junyanz/CycleGAN). It has been claimed to be faster and more accurate compared to CycleGAN, but I have not implemented and tested this approach yet.

### 3) ‫‪U-GAT-IT‬‬
According to the benchmarks, this approach is currently the best proposed approach for Unpaired Image-to-Image Translation. Unfortunately, due to the extremely lengthy training time, I have not been able to test this approach yet, but as soon as I have access to suitable hardware, I will add the code and results here.

### 4) Council-GAN
The results shown in the paper somehow claim better performance compared to the results obtained by U-GAT-IT‬‬. That is also on my todo list.


## Software and Hardware
I used the following specification:
- Cuda 10.1
- Tensorflow 2.3
- GTX 1080TI

## Contribution
Contributions are welcome and I'll be happy to recieve your help and advice with regard to implementations or other issues.
