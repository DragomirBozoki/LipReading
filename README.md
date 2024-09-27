# LipReadingProject
Deep Learning + Computer Vision + Natural Language Processing

The GRID corpus [33] is a comprehensive audiovisual database containing sentences spoken by 34 speakers (18 males and 16 females) in English. Each speaker utters 1,000 sentences, resulting in a total of 34,000 audio and video recordings. The sentences in the GRID corpus follow a specific structure, where each sentence adheres to the following pattern: command (4 options) + color (4 options) + preposition (4 options) + letter (25 options) + digit (10 options) + adverb (4 options). This approach allows for the generation of up to 64,000 unique sentences, such as: "set blue by four please" or "place red at C zero again."

![slika1](https://github.com/user-attachments/assets/aeb40ed1-8538-49f9-9212-50851dc2a226)

The data processing procedure begins with the analysis of each individual frame of the video, where the face in the video needs to be identified. The dlib library was used for face detection. After detecting the face (in the case of the GRID corpus, only one face is present in each video), landmarks defining the lips  are identified, and a 64x64 pixel region around the lips is cropped. Each frame is then converted to grayscale. The frames are grouped into a list, which is subsequently transformed into a tensor, standardized, and finally prepared for input into the neural network. An example of a prepared frame for the neural network is shown in the figure.

![slika2](https://github.com/user-attachments/assets/f07699a0-d8cf-4d4b-aa7b-f19662007f5a)
