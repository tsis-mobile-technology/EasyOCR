#-*-coding:utf-8-*-
"""
Copyright (c) 2019-present NAVER Corp.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os

import torch.nn as nn
import numpy as np
import yaml

from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import VGG_FeatureExtractor, RCNN_FeatureExtractor, ResNet_FeatureExtractor
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        # Modify User Code, EasyOCR + deep_text_recognition_benchmark
        #
        # def __init__(self, opt):
        # def __init__(self, input_channel, output_channel, hidden_size, num_class, FeatureExtraction, Prediction, SequenceModeling, Transformation, imgH, imgW, num_fiducial):
        # **network_params : input_channel, output_channel, hidden_size, num_class
        # ** user_params :  FeatureExtraction, Prediction, SequenceModeling, Transformation, imgH, imgW, num_fiducial
        #   FeatureExtraction='ResNet',
        #   Prediction='CTC',
        #   SequenceModeling='BiLSTM',
        #   Transformation='TPS',
        #   imgH=32,
        #   imgW=100,
        #   num_fiducial=20,
        # self.opt.input_channel = input_channel
        # self.opt.output_channel = output_channel
        # self.opt.hidden_size = hidden_size
        # self.opt.num_class = num_class
        # self.opt.FeatureExtraction = FeatureExtraction
        # self.opt.Prediction = Prediction
        # self.opt.SequenceModeling = SequenceModeling
        # self.opt.Transformation = Transformation
        # self.opt.imgH = imgH
        # self.opt.imgW = imgW
        # self.opt.num_fiducial = num_fiducial
        self.opt = opt
        print("opt.FeatureExtraction:", self.opt['FeatureExtraction'])
        print("opt.batch_max_length:", self.opt['batch_max_length'])
        # opt = self.opt
        self.stages = {'Trans': opt['Transformation'], 'Feat': opt['FeatureExtraction'],
                       'Seq': opt['SequenceModeling'], 'Pred': opt['Prediction']}

        """ Transformation """
        if opt['Transformation'] == 'TPS':
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt['num_fiducial'], I_size=(opt['imgH'], opt['imgW']), I_r_size=(opt['imgH'], opt['imgW']), I_channel_num=opt['input_channel'])
        else:
            print('No Transformation module specified')

        """ FeatureExtraction """
        if opt['FeatureExtraction'] == 'VGG':
            self.FeatureExtraction = VGG_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        elif opt['FeatureExtraction'] == 'RCNN':
            self.FeatureExtraction = RCNN_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        elif opt['FeatureExtraction'] == 'ResNet':
            self.FeatureExtraction = ResNet_FeatureExtractor(opt['input_channel'], opt['output_channel'])
        else:
            raise Exception('No FeatureExtraction module specified')
        self.FeatureExtraction_output = opt['output_channel']  # int(imgH/16-1) * 512
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))  # Transform final (imgH/16-1) -> 1

        """ Sequence modeling"""
        if opt['SequenceModeling'] == 'BiLSTM':
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(self.FeatureExtraction_output, opt['hidden_size'], opt['hidden_size']),
                BidirectionalLSTM(opt['hidden_size'], opt['hidden_size'], opt['hidden_size']))
            self.SequenceModeling_output = opt['hidden_size']
        else:
            print('No SequenceModeling module specified')
            self.SequenceModeling_output = self.FeatureExtraction_output

        """ Prediction """
        if opt['Prediction'] == 'CTC':
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt['num_class'])
        elif opt['Prediction'] == 'Attn':
            self.Prediction = Attention(self.SequenceModeling_output, opt['hidden_size'], opt['num_class'])
        else:
            raise Exception('Prediction is neither CTC or Attn')

    # deep-text-recognition-benchmark에서는 is_train이 False임
    # 또한 아래처럼 is_train=True로 하면 에러가 남
    #   에러 : indexerror: index 58 is out of bounds for dimension 1 with size 58
    # def forward(self, input, text, is_train=True):
    def forward(self, input, text, is_train=False):
        """ Transformation stage """
        if not self.stages['Trans'] == "None":
            input = self.Transformation(input)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(input)
        visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = visual_feature.squeeze(3)

        """ Sequence modeling stage """
        if self.stages['Seq'] == 'BiLSTM':
            contextual_feature = self.SequenceModeling(visual_feature)
        else:
            contextual_feature = visual_feature  # for convenience. this is NOT contextually modeled by BiLSTM

        """ Prediction stage """
        if self.stages['Pred'] == 'CTC':
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            # print(self.opt['batch_max_length'])
            prediction = self.Prediction(contextual_feature.contiguous(), text, is_train, batch_max_length=self.opt['batch_max_length'])

        return prediction
