# Face2Bitmoji
In this project, I'm going to use state-of-the-art approaches to Unpaired Image-to-Image Translation to tackle the problem of transforming human faces to cartoon faces (Bitmoji Faces) and try to improve them if possible.

## Database
Due to the lack of an organised dataset for Bitmoji faces, I made my own [Dataset](https://www.kaggle.com/mostafamozafari/bitmoji-faces?select=BitmojiDataset). It consists of 4084 Bitmojies. For human faces, I used [CelebAMask-HQ Dataset](https://github.com/switchablenorms/CelebAMask-HQ) and extracted full-frontal faces by the help of pose estimation model [FSA-NET](https://github.com/shamangary/FSA-Net).

## LAcycleGAN
This approach is based on [CycleGAN](https://github.com/junyanz/CycleGAN) and the authors have tried to improve CycleGAN by adding some parts to the model to improve the Geometric Transformations. The authors of the original paper have not published the codes for the model, so I tried to implement and slightly change them. There are some deficiencies in the current code but the results that it delivers are approximately the same as those of the original paper. In my experience, the suggested model is very hard to train and isn't stable enough to be considered as a good approach to this problem.
  


## Software and Hardware
I used the following specification:
- Cuda 10.1
- Tensorflow 2.3
- GTX 1080TI

## Contribution
Contributions are welcome and I'll be happy to recieve your help and advice with regard to implementations or other issues.
