# Img2VecCosSim-Django-Pytorch

Extract a feature vector for any image and find the cosine similarity for comparison using Pytorch. I have used ResNet-18 to extract the feature vector of images. Finally a Django app is developed to input two images and to find the cosine similarity.

#### Packages:
  - Pytorch
  - Django 2.0
  
#### How to start:
  * Clone repository
  
    `git clone https://github.com/MexsonFernandes/Img2VecCosSim-Django-Pytorch`
  * Change directory
  
    `cd Img2VecCosSim-Django-Pytorch`
  * Install virtual environment
  
    `pipenv install`
  * Install all dependencies
  
    `pipenv install -r requirements.txt` or `pip install -r requirements.txt`
    
  * Start django server
  
    `python manage.py runserver`

#### Showcase:
*Dashboard*
![Dashboard](https://raw.githubusercontent.com/realSaddy/Img2VecCosSim-Django-Pytorch/master/dashboard.png)

---

*Example Comparison*
![Example Comparison](https://github.com/realSaddy/Img2VecCosSim-Django-Pytorch/blob/master/example_comparison.png?raw=true)


#### Credits:
  - Inspired from <a href="https://becominghuman.ai/extract-a-feature-vector-for-any-image-with-pytorch-9717561d1d4c">Medium post by Christian Safka</a>
