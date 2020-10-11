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

scrp,gpu,bit,batch,dataset,iter = argv

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



####just for test
def calc(state):
	(state_o,state_p,state_n) = state	
	
	dis1 = state_o-state_p
	dis1 = dis1.pow(2)

	dis1 = dis1
	dis1 = dis1.sum(1)
	
	dis2 = state_o-state_n
	dis2 = dis2.pow(2)

	
	dis2 = dis2
	dis2 = dis2.sum(1)
	
	
	return dis1-dis2
	

####dataset
flag = True

if dataset=='cifar':
	traintest=Dataloader("/home/zhangjian/workspace/dataset/cifar10/resize64-rf-noncrop/train",0,500,0,10)
	Dtest=Dataloader("/home/zhangjian/workspace/dataset/cifar10/resize64-rf-noncrop/test",0,5500,0,10)
	flag = False

if dataset=='nus':
	traintest=Dataloader("/home/zhangjian/workspace/dataset/NUS-WIDE/resize64-rf-noncrop/train",0,500,1,22)
	flag = False

if dataset=='flk':
	traintest=Dataloader("None",0,500)
	flag = False

if flag:
	print('undefined_dataset')
	quit()

	
###model
model = ActorCritic(bit_len,batch_size,False)
model.load_state_dict(torch.load(checkpoint_path+'/'+iter+'.model'))
model.cuda()
print('model over')
episode_length = int(iter)
for i in range(episode_length):
	if i%steps==0:
		model.low_lr(rate)
###train

while True:
	episode_length += 1
	
	if episode_length%steps==0:
		model.low_lr(rate)
	
	if episode_length%1000==0:
		if dataset=='cifar':
			model.eval()
			map = test_util.test(Dtest,model,batch_size,bit_len)
			file=open(logpath,"a")
			file.write('#### map='+str(map)+'\n')
			file.close()
		path=checkpoint_path+'/'+str(episode_length)+'.model';
		torch.save(model.state_dict(),path)
		
	model.train()	
	
	states_o = []
	states_p = []
	states_n = []	
	if dataset=='flk':
		ori,pos,neg=traintest.get_batch_flk(batch_size)
	else:
		ori,pos,neg=traintest.get_batch_cifar_nus(batch_size)
		
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
	#	loss = F.triplet_margin_loss(probs1[i]-pb1,probs2[i]-pb2,probs3[i]-pb3)		
	#	print probs1[i]
		
		hash_o = torch.cat([hash_o,probs_o[i]],1)
		hash_p = torch.cat([hash_p,probs_p[i]],1)
		hash_n = torch.cat([hash_n,probs_n[i]],1)
		

	hash_o = hash_o[:,1:]
	hash_p = hash_p[:,1:]
	hash_n = hash_n[:,1:]
	
	final_loss = F.triplet_margin_loss(hash_o,hash_p,hash_n)

	if episode_length%20==0:
		if observe:
			print hash_o
			rit = 0
			tmp = calc((hash_o,hash_p,hash_n)).data
			tmp = tmp.cpu()
#			print tmp
			for i in range(batch_size):
				if tmp[i][0]<0:
					rit +=1
#					print (tmp[i][0],tmp[i][0]<0)
			print(rit,'/',batch_size)
		
		print(episode_length,final_loss.data[0])
		file=open(logpath,"a")
		file.write(str(episode_length)+' '+str(final_loss.data[0])+"\n")
		file.close()
	
#### updata the parameters

	model.zero_grad()
	
	final_loss[0].backward()
		
	model.step()

        
        