import math
import os
import sys

import torch
import torch.nn.functional as F
import torch.optim as optim
from model import ActorCritic
from torch.autograd import Variable
from dataloader import Dataloader


def get_ele(elem):
    return elem[0]    
	
def calc(a,b):
	for ix in range(len(a)):
		if a[ix]>0.5:
			a[ix]=1
		else:
			a[ix]=0
		if b[ix]>0.5:
			b[ix]=1
		else:
			b[ix]=0
	dist = a - b
#	print dist
	dist = dist.pow(2)
	dist = dist.sum()
	
	return dist

def get_map(query,data,query_lab,data_lab):
	tot = 0
	print 
	for i in range(len(query)):
		list = []
		eletot = 0.0
		cnt = 0
		for j in range(len(data)):
			list.append(  (calc(query[i],data[j]),data_lab[j])  )
		list.sort(key=get_ele)
		for j in range(len(list)):
			if list[j][1]==query_lab[i]:
				cnt += 1
				eletot += cnt*1.0/(1.0+j)
		eletot /= cnt
		tot += eletot
	tot /= len(query)
	return tot
	
def get_hash_and_label(hash,lab,batch_size,bit_len,size,data,model):
	for i in range(10):
		for ix in range(size):		
			pic = data.get_valid(i,batch_size,(3*ix+1)*batch_size)	
			pic=Variable(pic).cuda()		
			state = Variable(torch.zeros(batch_size,1).cuda())	
			probs = model(pic)
			for j in range(bit_len):		
				state = torch.cat([state,probs[j]],1)
			for j in range(batch_size):
				hash.append(state.data[j].cpu())
				lab.append(i)	
	
def test(data,model,batch_size,bit_len):
	model.eval()
	query = []
	q_lab = []
	datas = []
	d_lab = []
	get_hash_and_label(query,q_lab,batch_size,bit_len,1,data,model)
	get_hash_and_label(datas,d_lab,batch_size,bit_len,30,data,model)
	map = get_map(query,datas,q_lab,d_lab)
	return map
	


		

			
