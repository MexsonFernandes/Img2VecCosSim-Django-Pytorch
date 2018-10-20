import os, datetime, random

from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.shortcuts import render

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms

from torch.autograd import Variable
from PIL import Image
from FeatureVector.settings import MEDIA_ROOT

def handle_uploaded_file(f):
    name = str(datetime.datetime.now().strftime('%H%M%S')) + str(random.randint(0, 1000)) + str(f)
    path = default_storage.save(MEDIA_ROOT + '/' + name,
                                ContentFile(f.read()))
    return os.path.join(MEDIA_ROOT, path), name

def index(request):
    if request.POST:
        imgtovec = Img2Vec()
        file1_path, file1_name = handle_uploaded_file(request.FILES['file1'])
        file2_path, file2_name = handle_uploaded_file(request.FILES['file2'])
        pic_one_vector = imgtovec.get_vec(Image.open(file1_path))
        pic_two_vector = imgtovec.get_vec(Image.open(file2_path))
        # Using PyTorch Cosine Similarity
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        cos_sim = cos(torch.tensor(pic_one_vector).unsqueeze(0), torch.tensor(pic_two_vector).unsqueeze(0))

        print('\nCosine similarity: {0:.2f}\n'.format(float(cos_sim)))
        return render(request, "index.html", {"cos_sim": 'Score: {0:.2f}'.format(float(cos_sim)),
                                              "post": True,
                                              "img1src": file1_name,
                                              "img2src": file2_name,
                                              })
    return render(request, "index.html", {'post': False})


class Img2Vec:
    def __init__(self, cuda=True, model='resnet-18', layer='default', layer_output_size=512):
        """ Img2Vec
        :param cuda: If set to True, will run forward pass on GPU
        :param model: String name of requested model
        :param layer: String or Int depending on model.  See more docs: https://github.com/christiansafka/img2vec.git
        :param layer_output_size: Int depicting the output size of the requested layer
        """
        self.device = torch.device("cuda" if cuda else "cpu")
        self.layer_output_size = layer_output_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()

        self.scaler = transforms.Scale((224, 224))
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                              std=[0.229, 0.224, 0.225])
        self.to_tensor = transforms.ToTensor()

    def get_vec(self, img, tensor=False):
        """ Get vector embedding from PIL image
        :param img: PIL Image
        :param tensor: If True, get_vec will return a FloatTensor instead of Numpy array
        :returns: Numpy ndarray
        """
        image = self.normalize(self.to_tensor(self.scaler(img))).unsqueeze(0).to(self.device)

        my_embedding = torch.zeros(1, self.layer_output_size, 1, 1)

        def copy_data(m, i, o):
            my_embedding.copy_(o.data)

        h = self.extraction_layer.register_forward_hook(copy_data)
        h_x = self.model(image)
        h.remove()

        if tensor:
            return my_embedding
        else:
            return my_embedding.numpy()[0, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


