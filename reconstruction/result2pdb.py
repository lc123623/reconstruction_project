import voxel2pdb
import pdb2voxel
import align
import os
import numpy as np
import auto_encoder



def write2pdb(group,rmax,output_folder):
	num=len(group)
	os.system('mkdir %s/sub2'%output_folder)
	os.system('mkdir %s/sub3'%output_folder)
	for ii in range(num):
		voxel2pdb.write_pdb(group[ii],'%s/sub2/%d.pdb'%(output_folder,ii),rmax)
	fix='%s/sub2/0.pdb'%output_folder
	data=[]
	for ii in range(num):
		mov='%s/sub2/%d.pdb'%(output_folder,ii)
		align.run(fix,mov,'%s/sub3/%d.pdb'%(output_folder,ii))
		voxel=pdb2voxel.run(['pdbfile=%s/sub3/%d.pdb'%(output_folder,ii)])
		data.append(voxel)
	data=np.array(data)
	data=np.mean(data,axis=0)
	data=np.greater(data,0.5).astype(int)
	voxel2pdb.write_pdb(data,'%s/out.pdb'%output_folder,rmax)
	os.system('rm -rf %s/sub2'%output_folder)
	os.system('rm -rf %s/sub3'%output_folder)

def cal_cc(voxel_group,rmax,output_folder,target_pdb):
	os.system('mkdir %s/temp'%output_folder)
	num=voxel_group.shape[0]
	cc_mat=np.zeros(shape=(num,20))
	for ii in range(num):
		for jj in range(20):
			voxel2pdb.write_pdb(voxel_group[ii,jj],'%s/temp/%d_%d.pdb'%(output_folder,ii,jj),rmax)
			cc=align.run(fix=target_pdb,mov='%s/temp/%d_%d.pdb'%(output_folder,ii,jj))
			cc_mat[ii,jj]=cc 
			print ii,jj,'%.3f'%cc
	np.save('%s/cc_mat.npy'%output_folder,cc_mat)
	np.savetxt('%s/cc_mat.txt'%output_folder,cc_mat,fmt=='%.3f')
	os.system('rm -rf %s/temp'%output_folder)

	
