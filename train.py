#coding:utf-8
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
import test_util
from sys import argv
import function

scrp,gpu,bit,batch,dataset = argv

steps = 10000
rate = 0.1
observe = True

##setting
checkpoint_path = 'checkpoint-%s-%sbit' % (dataset,bit)
logpath = 'log-%s-%sbit.txt' % (dataset,bit)
if not os.path.exists(checkpoint_path):
	os.mkdir(checkpoint_path)
os.environ['CUDA_VISIBLE_DEVICES'] = gpu
batch_size = int(batch)
bit_len = int(bit)

####dataset
flag = True

if dataset=='cifar':
	Dtest=Dataloader("/home/yezhaoda/TMM-DRLIH/cifar10/resize64-rf-noncrop/test",0,5500,0,10)
	traintest=Dataloader("/home/yezhaoda/TMM-DRLIH/cifar10/resize64-rf-noncrop/train",0,500,0,10)
	flag = False

if dataset=='nus':
	traintest=Dataloader("/home/zhangjian/workspace/dataset/NUS-WIDE/resize64-rf-noncrop/train",0,500,1,22,'nus')
	flag = False

if dataset=='flk':
	traintest=Dataloader("None",0,500,1,1,'flk')
	flag = False

if flag:
	print('undefined_dataset')
	quit()

	
###model
model = ActorCritic(bit_len,batch_size)
model.cuda()
print('model over')

###train

episode_length = 0
while True:
	episode_length += 1
	
	
	if episode_length%steps==0:
		model.low_lr(rate)
	
	if episode_length%1000==0:
		if dataset=='cifar':
			model.eval()
#			map = test_util.test(Dtest,model,batch_size,bit_len)
			file=open(logpath,"a")
			file.write('#### map='+str(map)+'\n')
			file.close()
		path=checkpoint_path+'/'+str(episode_length)+'.model';
		torch.save(model.state_dict(),path)
		
	model.train()	
	

	
	if dataset=='cifar':
		ori,pos,neg=traintest.get_batch_cifar_nus(batch_size)
	else:
		ori,pos,neg=traintest.get_batch_flk_nus(batch_size)		
		
	
	ori=Variable(ori).cuda()
	pos=Variable(pos).cuda()
	neg=Variable(neg).cuda()


	hash_o = Variable(torch.zeros(batch_size,1).cuda())
	hash_p = Variable(torch.zeros(batch_size,1).cuda())
	hash_n = Variable(torch.zeros(batch_size,1).cuda())
	probs_o = model(ori)
	probs_p = model(pos)
	probs_n = model(neg)
	

	for i in range(bit_len):			
		hash_o = torch.cat([hash_o,probs_o[i]],1)
		hash_p = torch.cat([hash_p,probs_p[i]],1)
		hash_n = torch.cat([hash_n,probs_n[i]],1)
		
	hash_o = hash_o[:,1:]
	hash_p = hash_p[:,1:]
	hash_n = hash_n[:,1:]
	
#### loss
	tri_loss = function.triplet_margin_loss(hash_o,hash_p,hash_n)
	
	tmp_prob = (function.log_porb(hash_o))/(bit_len)
	loss_L = torch.mean(tmp_prob * tri_loss)
	
	loss_R = torch.mean(tri_loss)
	
	final_loss = 0.05*loss_L + 0.95*loss_R

#### 更新参数

	model.zero_grad()
	
	final_loss.backward()
		
	model.step()
	
	
	if episode_length%20==0:
		print(str(episode_length)+' '+str(final_loss.item())+" "+str(loss_L.item())+" "+str(loss_R.item())+"\n")
		file=open(logpath,"a")
		file.write(str(episode_length)+' '+str(final_loss.item())+" "+str(loss_L.item())+" "+str(loss_R.item())+"\n")
		file.close()
        