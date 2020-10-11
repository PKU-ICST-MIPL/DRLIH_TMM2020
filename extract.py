import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from torch.autograd import Variable
from torchvision import datasets, transforms
from dataloader import Dataloader
import math
from sys import argv


script,gpu,bit,batch,dataset,iter = argv

##setting
checkpoint_path = 'checkpoint-%s-%sbit' % (dataset,bit)
fea_path = '%s-%sbit-fea.txt' % (dataset,bit)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
batch_size = int(batch)
bit_len = int(bit)

####dataset
flag = True
size = 0
if dataset=='cifar':
	Dtest=Dataloader("/home/yezhaoda/TMM-DRLIH/cifar10/resize64-rf-noncrop/test",0,5500,0,10)
	size = 55000
	flag = False

if dataset=='nus':
	Dtest=Dataloader("/home/zhangjian/workspace/dataset/NUS-WIDE/resize64-rf-noncrop/test",0,155547)
	size = 155547
	flag = False

if dataset=='flk':
	Dtest=Dataloader("/home/zhangjian/workspace/dataset/MIRFlickr/resize64-rf-noncrop/test",0,20000)
	size = 20000
	flag = False

if flag:
	print('undefined_dataset')
	quit()
	

model = ActorCritic(bit_len,batch_size,False)
model.load_state_dict(torch.load(checkpoint_path+'/'+iter+'.model'))
model.cuda()
model.eval()

f_fea=open(fea_path,"w")
for ix in range(0,int(size/batch_size)):

	if dataset=='cifar':
		pic,lab=Dtest.get_test_cifar(batch_size)
	else:
		pic,lab=Dtest.get_test_nus_flk(batch_size)
	
	pic=Variable(pic).cuda()
	probs = model(pic)
	
	hash = torch.zeros(batch_size,1)
	for i in range(bit_len):
		hash = torch.cat([hash,probs[i].data.cpu()],1)
	hash = hash[:,1:]
	
	if (ix%100==0):
		print(ix*batch_size,'img load')
	
	for bx in range(batch_size):
		for i in range(bit_len):
			f_fea.write(str(hash[bx][i].item())+' ')
		f_fea.write("\n")
		
f_fea.close()
