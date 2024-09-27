# LipReadingProject
Deep Learning + Computer Vision + Natural Language Processing

The GRID corpus [33] is a comprehensive audiovisual database containing sentences spoken by 34 speakers (18 males and 16 females) in English. Each speaker utters 1,000 sentences, resulting in a total of 34,000 audio and video recordings. The sentences in the GRID corpus follow a specific structure, where each sentence adheres to the following pattern: command (4 options) + color (4 options) + preposition (4 options) + letter (25 options) + digit (10 options) + adverb (4 options). This approach allows for the generation of up to 64,000 unique sentences, such as: "set blue by four please" or "place red at C zero again."

![slika1](https://github.com/user-attachments/assets/aeb40ed1-8538-49f9-9212-50851dc2a226)

The data processing procedure begins with the analysis of each individual frame of the video, where the face in the video needs to be identified. The dlib library was used for face detection. After detecting the face (in the case of the GRID corpus, only one face is present in each video), landmarks defining the lips  are identified, and a 64x64 pixel region around the lips is cropped. Each frame is then converted to grayscale. The frames are grouped into a list, which is subsequently transformed into a tensor, standardized, and finally prepared for input into the neural network. An example of a prepared frame for the neural network is shown in the figure.

![slika2](https://github.com/user-attachments/assets/f07699a0-d8cf-4d4b-aa7b-f19662007f5a)

Before the processed data is passed to the model, it must be organized into the appropriate dataset. This process begins by identifying the path to the folder containing the video recordings. After that, the data (video recordings) are randomly shuffled to reduce the risk of overfitting. Next, the data are aligned with the corresponding labels for each frame. The data are grouped into batches of size 2, with each batch padded to dimensions of ([75, 64, 64], [40]), where 75 represents the length of the video, 64x64 corresponds to the width and height of the frames, and 40 is the number of possible characters associated with each frame. Prefetching is used to optimize performance by asynchronously loading data in advance. The data are then split into training and test sets: the first 450 video recordings are used for training the model, while the remaining 50 videos are used for validating the algorithm.

![model](https://github.com/user-attachments/assets/d019d108-8b58-4907-b340-08813a53ec20)

The CTC loss function is used for training the neural network. To optimally monitor the training process, various callback functions have been implemented. Among them is a scheduler that maintains a constant learning rate during the first 30 epochs, after which it decays exponentially, allowing the network to converge more effectively.

The Adam optimizer, known for its efficiency and robustness in gradient-based learning, is used for model optimization, with the initial learning rate set to 0.0001. The training process spans 450 epochs, changing speakers each 50 epoches, during which the network gradually adjusts to achieve the best possible performance on the task while avoiding overfitting.
