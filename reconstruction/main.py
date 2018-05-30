import numpy as np 
import auto_encoder 
import exceptions
import map2iq
import time
import multiprocessing
import threading
import region_search
import os
import sys
import result2pdb
import argparse
np.set_printoptions(precision=3,threshold=np.NaN)

parser=argparse.ArgumentParser()
parser.add_argument('--iq_path',help='path of iq_file',type=str)
parser.add_argument('--rmax',help='radius of the protein',type=float)
parser.add_argument('--output_folder',help='path of output file',type=str)
parser.add_argument('--target_pdb',help='path of target pdb file',default='None',type=str)
args=parser.parse_args()

class MyThread(threading.Thread):
	def __init__(self,func,args=()):
		super(MyThread,self).__init__()
		self.func=func
		self.args=args

	def run(self):
		self.result=self.func(*self.args)

	def get_result(self):
		try:
			return self.result
		except Exception:
			return None

class evolution:

	def __init__(self,original_group,output_folder,gpu_num=10):

		
		self.output_folder=output_folder
		self.gpu_num=gpu_num

		self.iteration_step=0
		self.counter=0

		self.gene_length=200
		self.exchange_gene_num=1
		self.group_num=300
		self.inheritance_num=300
		self.remain_best_num=2
		self.statistics_num=20

		self.compute_score=map2iq.run
		self.sess,self.in_tensor,self.z_tensor,self.out_tensor=auto_encoder.generate_session(self.gpu_num)
		print 'generate computing graph'

		original_num=original_group.shape[0]
		self.original_group=original_group
		self.original_group_score=self.compute_group_score(original_group)
		self.original_group,self.original_group_score=self.rank_group(self.original_group,self.original_group_score)
		
		self.group=np.copy(self.original_group[:self.group_num])
		self.group_score=np.copy(self.original_group_score[:self.group_num])

		self.best_so_far=np.copy(self.original_group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.original_group_score[:self.remain_best_num])

		self.score_mat=self.group_score[:self.statistics_num].reshape((1,self.statistics_num))
		self.gene_data=np.copy(self.group[:self.statistics_num]).reshape((1,self.statistics_num,200))

		print 'original input , top5:',self.group_score[:5]
		print 'mean_score is:',np.mean(self.group_score)
		print 'initialized'
	
	def encode(self,cube,n):
		in_=np.zeros(shape=(1,32,32,32,1))
		in_[0,:31,:31,:31,0]=cube
		z_,out_=self.sess.run([self.z_tensor[n],self.out_tensor[n]],feed_dict={self.in_tensor[n]:in_})
		z_=z_.reshape((self.gene_length))
		out_=np.greater(out_,0.5).astype(int)
		out_=out_[0,:31,:31,:31,0].reshape((31,31,31))
		return z_,out_

	def encode_group(self,cube_group,n):
		num=cube_group.shape[0]
		group_z=np.zeros(shape=(num,self.gene_length))
		group_out=np.zeros(shape=(num,31,31,31))
		for ii in range(num):
			group_z[ii],group_out[ii]=self.encode(cube_group[ii],n)
		return [group_z,group_out]

	def multi_thread_encode_group(self,cube_group):

		num=cube_group.shape[0]
		sub_group_num=num//(self.gpu_num-1)
		threads=[]
		for ii in range(self.gpu_num-1):
			t=MyThread(self.encode_group,args=(cube_group[(ii*sub_group_num):((ii+1)*sub_group_num)],ii))
			threads.append(t)
			t.start()
		t=MyThread(self.encode_group,args=(cube_group[((self.gpu_num-1)*sub_group_num):],(self.gpu_num-1)))
		threads.append(t)
		t.start()
		z_group=[]
		for t in threads:
			t.join()
		z_group=[]
		real_data_group=[]
		for t in threads:
			result=t.get_result()
			real_data_group.append(result[1])
			z_group.append(result[0])
		z_group=np.concatenate(z_group,axis=0)
		real_data_group=np.concatenate(real_data_group,axis=0)
		
		return z_group,real_data_group
	

	def decode(self,gene,n):
		gene=gene.reshape((1,self.gene_length))	
		real_data=self.sess.run(self.out_tensor[n],feed_dict={self.z_tensor[n]:gene})
		real_data=np.greater(real_data,0.5).astype(int)
		real_data=real_data[0,:31,:31,:31,0].reshape((31,31,31))
		return real_data

	def decode_group(self,gene_group,n):
		num=gene_group.shape[0]
		group_real_data=np.zeros(shape=(num,31,31,31))
		for ii in range(num):
			group_real_data[ii]=self.decode(gene_group[ii],n)
		return group_real_data

	def multi_thread_decode_group(self,group):

		num=group.shape[0]
		sub_group_num=num//(self.gpu_num-1)
		threads=[]
		for ii in range(self.gpu_num-1):
			t=MyThread(self.decode_group,args=(group[(ii*sub_group_num):((ii+1)*sub_group_num)],ii))
			threads.append(t)
			t.start()
		if (self.gpu_num-1)*sub_group_num!=num:
			t=MyThread(self.decode_group,args=(group[((self.gpu_num-1)*sub_group_num):],(self.gpu_num-1)))
			threads.append(t)
			t.start()
		real_data_group=[]
		for t in threads:
			t.join()
		for t in threads:
			sub_real_data_group=t.get_result()
			real_data_group.append(sub_real_data_group)
		real_data_group=np.concatenate(real_data_group,axis=0)
		
		return real_data_group

	def region_process(self,cube_group,index):
		num=cube_group.shape[0]
		z_group=[]
		real_data_group=[]
		for ii in range(num):
			out=cube_group[ii]
			while True:
				in_=np.zeros(shape=(1,32,32,32,1))
				in_[0,:31,:31,:31,0]=out
				z_,out_=self.sess.run([self.z_tensor[index],self.out_tensor[index]],feed_dict={self.in_tensor[index]:in_})
				z_=z_.reshape((self.gene_length))
				real_data=np.greater(out_[0,:31,:31,:31,0].reshape((31,31,31)),0.5).astype(int)
				out,region_num=region_search.find_biggest_region(real_data)
				if region_num<=1:
					break
			z_group.append(z_.reshape(1,self.gene_length))
			real_data_group.append(real_data.reshape(1,31,31,31))
		z_group=np.concatenate(z_group,axis=0)
		real_data_group=np.concatenate(real_data_group,axis=0)
		return [z_group,real_data_group]

	def compute_group_score(self,group):
		
		real_data_group=self.multi_thread_decode_group(group)
		num=group.shape[0]
		
		t1=time.time()
		region_inf=np.empty(shape=(num),dtype=np.bool)

		pool=multiprocessing.Pool(processes=20)
		result=pool.map(region_search.find_biggest_region,real_data_group)
		pool.close()
		pool.join()
		for ii in range(num):
			if result[ii][1]>1:
				region_inf[ii]=True
				real_data_group[ii]=result[ii][0]
			else:
				region_inf[ii]=False

		data_to_process=real_data_group[region_inf]
		#print 'num to encode:',data_to_process.shape[0]
		data_unchanged=real_data_group[(1 - region_inf).astype(bool)]
		z_unchanged=group[(1 - region_inf).astype(bool)]

		t2=time.time()
		#print 'find region time:',t2-t1

		if data_to_process.shape[0]==0:
			real_data_group=data_unchanged
			self.group=z_unchanged
		
		elif (data_to_process.shape[0]<10) & (data_to_process.shape[0]>0):
			result=self.region_process(data_to_process,0)
			data_processed=result[1]
			z_processed=result[0]
			real_data_group=np.concatenate([data_unchanged,data_processed],axis=0)
			self.group=np.concatenate([z_unchanged,z_processed],axis=0)
		else:
			threads=[]
			sub_group_num=data_to_process.shape[0]//10
			for ii in range(9):
				t=MyThread(self.region_process,args=(data_to_process[(ii*sub_group_num):((ii+1)*sub_group_num)],ii))
				threads.append(t)
				t.start()
			t=MyThread(self.region_process,args=(data_to_process[(9*sub_group_num):],9))
			threads.append(t)
			t.start()
			data_processed=[]
			z_processed=[]
			for t in threads:
				t.join()
			for t in threads:
				result=t.get_result()
				data_processed.append(result[1])
				z_processed.append(result[0])
			data_processed=np.concatenate(data_processed,axis=0)
			z_processed=np.concatenate(z_processed,axis=0)
			real_data_group=np.concatenate([data_unchanged,data_processed],axis=0)
			self.group=np.concatenate([z_unchanged,z_processed],axis=0)
		pool=multiprocessing.Pool(processes=20)
		result=pool.map(self.compute_score,real_data_group)
		pool.close()
		pool.join()
		group_score=np.array(result)	
		return group_score


 
	def rank_group(self,group,group_score):
		index=np.argsort(group_score)
		group=group[index]
		group_score=group_score[index]
		return group,group_score
		
					

	def exchange_gene(self,selective_gene):
		np.random.shuffle(selective_gene)
		for ii in range(0,self.inheritance_num-self.remain_best_num,2):
			cross_point=np.random.randint(0,self.gene_length,size=(2*self.exchange_gene_num))
			cross_point=np.sort(cross_point)
			for jj in range(self.exchange_gene_num):
				random_data=np.random.uniform(low=0,high=1)
				if random_data<0.8:
					temp=np.copy(selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]])
					selective_gene[ii,cross_point[jj*2]:cross_point[jj*2+1]]=selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]			
					selective_gene[ii+1,cross_point[jj*2]:cross_point[jj*2+1]]=np.copy(temp)	


	def gene_variation(self,selective_gene):
		for ii in range(self.inheritance_num-self.remain_best_num):
			random_data=np.random.uniform(low=0,high=1,size=(self.gene_length))
			for jj in range(self.gene_length):
				if random_data[jj]<0.05:
					gene_point=np.random.randint(low=0,high=300)
					selective_gene[ii,jj]=gene_point

	
	def select_group(self):
		mixture=np.concatenate((self.group,self.group_score.reshape((-1,1))),axis=1)
		np.random.shuffle(self.group)
		self.group=mixture[:,:200]
		self.group_score=mixture[:,-1].reshape((-1))
		selected_group=np.zeros(shape=(self.inheritance_num-self.remain_best_num,self.gene_length))
		selected_group_score=np.zeros(shape=(self.inheritance_num-self.remain_best_num))
		for ii in range(self.inheritance_num-self.remain_best_num):
			a=np.random.randint(0,self.group_num)
			b=np.random.randint(0,self.group_num)
			if self.group_score[a]<self.group_score[b]:
				selected_group[ii]=np.copy(self.group[a])
				selected_group_score[ii]=np.copy(self.group_score[a])
			else:
				selected_group[ii]=np.copy(self.group[b])
				selected_group_score[ii]=np.copy(self.group_score[b])
		self.group=selected_group
		self.group_score=selected_group_score


		

	def inheritance(self):
		self.select_group()
		self.exchange_gene(self.group)
		self.gene_variation(self.group)
		if self.group.shape[0]!=self.inheritance_num-self.remain_best_num:
			raise Exception('bad')
		self.group_score=self.compute_group_score(self.group)
		self.group=np.concatenate((self.group,self.best_so_far),axis=0)
		self.group_score=np.concatenate((self.group_score,self.best_so_far_score),axis=0)

		self.group,self.group_score=self.rank_group(self.group,self.group_score)
		self.gene_data=np.concatenate((self.gene_data,self.group[:self.statistics_num].reshape((1,self.statistics_num,200))),axis=0)
		self.score_mat=np.concatenate((self.score_mat,self.group_score[:self.statistics_num].reshape((1,self.statistics_num))),axis=0)

		self.best_so_far=np.copy(self.group[:self.remain_best_num])
		self.best_so_far_score=np.copy(self.group_score[:self.remain_best_num])

		self.group=np.copy(self.group[:self.group_num])
		self.group_score=np.copy(self.group_score[:self.group_num])




	def evolution_iteration(self):
		while True:
			t1=time.time()
			self.inheritance()
			self.iteration_step=self.iteration_step+1
			t2=time.time()
			print self.score_mat.shape
			print 'iteration_step:',self.iteration_step,'top5:',self.group_score[:5],'mean_score is:',np.mean(self.score_mat[-1]),self.group_num
			
			if self.score_mat[-1,0]<self.score_mat[-2,0]:
				self.counter=0
			else:
				self.counter=self.counter+1
				if self.counter>20:
					self.group_num=self.group_num-100
					self.counter=0
					if self.group_num<100:
						np.save('%s/score_mat.npy'%self.output_folder,self.score_mat)
						np.savetxt('%s/score_mat.txt'%self.output_folder,self.score_mat,fmt=='%.3f')
						result_sample=self.multi_thread_decode_group(self.group)
						gene=self.gene_data.reshape((-1,200))
						voxel_group=self.multi_thread_decode_group(gene)
						voxel_group=voxel_group.reshape((-1,self.statistics_num,31,31,31))
						self.sess.close()
						return result_sample[:self.statistics_num],voxel_group



def generate_original_group(num):
	a=np.random.randint(low=0,high=5,size=(num*200))
	a=np.greater(a,0).astype(int)
	b=np.random.normal(150,100,size=(num*200))
	original_group=np.multiply(a,b).reshape((num,200))
	return original_group



if __name__=='__main__':
	iq_path=args.iq_path
	rmax=args.rmax
	output_folder=args.output_folder
	target_pdb=args.target_pdb

	cur_path=sys.path[0]
	map2iq.iq_path=iq_path
	map2iq.rmax=rmax
	auto_encoder.saved_model_path='/mnt/data2/liucan/protein_evolution/model'
	
	t1=time.time()
	original_group=generate_original_group(300)
	genetic_object=evolution(original_group,output_folder)
	result_sample,voxel_group=genetic_object.evolution_iteration()

	result2pdb.write2pdb( result_sample ,rmax ,output_folder)

	if target_pdb is not None:
		result2pdb.cal_cc(voxel_group,rmax,output_folder,target_pdb)
	t2=time.time()
	total_time=t2-t1
	print 'total_time:',total_time
	logfile=open('%s/log.txt'%output_folder,'a')
	logfile.write('total time: %d'%total_time)
	logfile.close()





