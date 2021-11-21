import torch
import random
from torchvision import transforms as trans
from PIL import Image

normalize = trans.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
								 
normalize2 = trans.Normalize(mean=[0.4465, 0.4123, 0.3725],
                                 std=[0.2406, 0.2312, 0.2283])
								 

class Dataloader(object):
	def __init__(self,path,start=0,end=500,class_st=0,class_ed=10,dataset='cifar'):
		self.path=path
		self.classst=class_st
		self.classed=class_ed
		self.li= list(range(self.classst,self.classed))
		self.trans1=trans.Compose([trans.CenterCrop(224),trans.RandomHorizontalFlip(),trans.ToTensor(),normalize2,])
		self.trans2=trans.Compose([trans.CenterCrop(224),trans.ToTensor(),])
		self.trans3=trans.Compose([trans.RandomCrop(224),trans.RandomHorizontalFlip(),trans.ToTensor(),normalize,])
		self.trans4=trans.Compose([trans.CenterCrop(224),trans.ToTensor(),normalize,])
		self.start=start
		self.end=end
		self.classidx=self.classst
		self.ix=start
		if dataset=='flk':
			fp=open("/home/zhangjian/workspace/dataset/MIRFlickr/tri_list.txt","r")
			self.tri_list=fp.readlines()
			self.tri_idx_max=199990
		elif dataset=='nus':
			fp=open("/home/zhangjian/workspace/dataset/NUS-WIDE/nus_tri_list.txt","r")
			self.tri_list=fp.readlines()
			self.tri_idx_max=500000
		self.tri_idx = 0
		
		
 
	def get_image(self,classid,imageid):
		image_path=self.path+'/'+str(classid)+'/'+str(imageid)+'.jpg';
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans3(ori_img)		
		return img
		
	def get_image2(self,classid,imageid):
		image_path=self.path+'/'+str(classid)+'/'+str(imageid)+'.jpg';
		print('get img',image_path)
		ori_img=Image.open(image_path)
#		ori_img=cv2.resize(ori_img,(224,224))
		img=self.trans2(ori_img)		
		return img
		
	def get_image3(self,classid,imageid):
		image_path=self.path+'/'+str(classid)+'/'+str(imageid)+'.jpg';
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans4(ori_img)		
		return img
		
	def get_image4(self,imageid):
		image_path=self.path+'/'+str(imageid)+'.jpg';
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans4(ori_img)		
		return img
		
	def get_image5(self,classid,imageid):
		image_path=self.path+'/'+str(classid)+'/'+str(imageid)+'.jpg';
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans4(ori_img)		
		return img

	def get_image_path(self,image_path):
#		print('get img',image_path)
		ori_img=Image.open(image_path)
		img=self.trans4(ori_img)		
		return img
		
	def get_batch_cifar_nus(self,batch_size=5):		
		li1= list(range(self.start,self.end))
		ori=torch.zeros(batch_size,3,224,224)
		pos=torch.zeros(batch_size,3,224,224)
		neg=torch.zeros(batch_size,3,224,224)
		for ix in range(0,batch_size):
			random.shuffle(li1)
			random.shuffle(self.li)
			ori[ix]=self.get_image(self.li[0],li1[0])
			pos[ix]=self.get_image(self.li[0],li1[1])
			neg[ix]=self.get_image(self.li[1],li1[2])
		return ori,pos,neg
		
		
	def get_batch(self,batch_size=5):
		random.shuffle(self.li)
		li1=range(self.start,self.end)
		ori=torch.zeros(batch_size,3,224,224)
		pos=torch.zeros(batch_size,3,224,224)
		neg=torch.zeros(batch_size,3,224,224)
		for ix in range(0,batch_size):
			random.shuffle(li1)
			ori[ix]=self.get_image(self.li[0],li1[0])
			pos[ix]=self.get_image(self.li[0],li1[1])
			neg[ix]=self.get_image(self.li[1],li1[2])
		return ori,pos,neg
		
	def get_batch_flk_nus(self,batch_size=5):
		ori=torch.zeros(batch_size,3,224,224)
		pos=torch.zeros(batch_size,3,224,224)
		neg=torch.zeros(batch_size,3,224,224)
		for ix in range(0,batch_size):
			path = self.tri_list[self.tri_idx].replace("\n",'').replace("\r",'')
			path = path.split(' ')
			ori[ix]=self.get_image_path(path[0])
			pos[ix]=self.get_image_path(path[1])
			neg[ix]=self.get_image_path(path[2])
			self.tri_idx += 1
			if self.tri_idx>self.tri_idx_max:
				self.tri_idx = 0
		return ori,pos,neg
		
		
	def get_test_nus_flk(self,batch_size=16):
		pic=torch.zeros(batch_size,3,224,224)
		label=torch.zeros(batch_size,1).long()
		for ix in range(0,batch_size):
			pic[ix]=self.get_image4(self.ix)
			label[ix]=0
			self.ix+=1
			if self.ix==self.end:
				self.ix=self.start
		return pic,label
		
	def get_test_cifar(self,batch_size=16):
		pic=torch.zeros(batch_size,3,224,224)
		label=torch.zeros(batch_size,1)
		for ix in range(0,batch_size):
			pic[ix]=self.get_image3(self.classidx,self.ix)
			label[ix]=self.classidx
			self.ix+=1
			if self.ix==self.end:
				self.ix=self.start
				self.classidx+=1
			if self.classidx==self.classed:
				self.classidx=self.classst
		return pic,label

		
	def get_train(self,batch_size=16):
		pic=torch.zeros(batch_size,3,224,224)
		label=torch.zeros(batch_size,1)
		for ix in range(0,batch_size):
			pic[ix]=self.get_image5(self.classidx,self.ix)
			label[ix]=self.classidx
			self.ix+=1
			if self.ix==self.list[self.classidx]:
				self.ix=self.start
				self.classidx+=1
			if self.classidx==self.classed:
				self.classidx=0
		return pic,label		

	def get_valid(self,classidx,batch_size,bias):
		pic=torch.zeros(batch_size,3,224,224)
		for ix in range(bias,bias+batch_size):
			pic[ix-bias]=self.get_image3(classidx,ix)
		return pic
		
			
