import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as md
import torch.optim as optim
def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1).expand_as(out))
    return out

class ActorCritic(torch.nn.Module):

	def __init__(self,bit_len,batch_size,pre=True):
		super(ActorCritic, self).__init__()
		
		self.bit_len = bit_len
		self.batch_size = batch_size
		
		self.model=md.vgg19_bn(False)				
		newmodel = nn.Sequential( nn.Linear(25088,4096),nn.ReLU(inplace=True),nn.Dropout(p=0.5),nn.Linear(4096,4096))				
		self.model.classifier = newmodel
		
		if pre:
			model_dict = self.model.state_dict()
			ori_model=md.vgg19_bn(True)
			ori_dict = ori_model.state_dict()
			pre_dict = {k:v for k,v in ori_dict.items() if k in model_dict}
			model_dict.update(pre_dict)
			self.model.load_state_dict(model_dict)
		
#		self.fet = nn.Sequential( nn.Linear(4096,256),nn.ReLU(inplace=True),nn.Dropout(p=0.5))
		
		self.lstm = nn.RNNCell(4096, 4096)
		self.lstm.bias_ih.data.fill_(0)
		self.lstm.bias_hh.data.fill_(0)
		
		self.actor_linear = nn.Linear(4096, 1)				
		self.actor_linear.weight.data = normalized_columns_initializer(
			self.actor_linear.weight.data, 0.01)
		self.actor_linear.bias.data.fill_(0)
		
		self.opt1 = optim.SGD(self.model.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt2 = optim.SGD(self.lstm.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
		self.opt3 = optim.SGD(self.actor_linear.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)
#		self.opt4 = optim.SGD(self.fet.parameters(), lr=0.001,momentum=0.9, weight_decay=0.0005)


	def forward(self, inputs):
		
		fc6 = self.model(inputs)
		hx = fc6
		probs = []
		for step in range(self.bit_len):
			hx = self.lstm(hx, hx)
			prob = torch.sigmoid(self.actor_linear(hx))
			probs.append(prob)
        
		return probs
	
	def zero_grad(self):
		self.opt1.zero_grad()
		self.opt2.zero_grad()
		self.opt3.zero_grad()

	def step(self):
		self.opt1.step()
		self.opt2.step()
		self.opt3.step()


	def low_lr(self,rate):
		for pg in self.opt1.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt2.param_groups:
			pg['lr'] = pg['lr'] * rate
			
		for pg in self.opt3.param_groups:
			pg['lr'] = pg['lr'] * rate
					
