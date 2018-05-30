import numpy as np
import matplotlib.pyplot as plt

'''
target_pdb=sys.argv[1]
rmax=float(sys.argv[2])
saved_model_path=sys.argv[3]
result_folder=sys.argv[4]
iq_file=sys.argv[5]

auto_encoder.saved_model_path=saved_model_path
sess,in_tensor,z_tensor,out_tensor=auto_encoder.generate_session(10)

data=np.load('%s/best_gene.npy'%result_folder)
num=data.shape[0]
print num
score_mat=np.zeros(shape=(num,20))
cc_mat=np.zeros(shape=(num,20))

os.system('mkdir %s/temp'%result_folder)

for ii in range(num):
	for jj in range(20):

		gene=data[ii,jj,:].reshape((1,-1))
		out_=sess.run(out_tensor[0],feed_dict={z_tensor[0]:gene})
		out_=np.greater(out_,0.5).astype(int)
		out_=np.reshape(out_[:,:31,:31,:31,:],(31,31,31))
		score_mat[ii,jj]=map2iq.run(out_,iq_file,50)
		voxel2pdb.write_pdb(out_,'%s/temp/%d_%d.pdb'%(result_folder,ii,jj),rmax)
		cc=align.run(fix=target_pdb,mov='%s/temp/%d_%d.pdb'%(result_folder,ii,jj))
		cc_mat[ii,jj]=cc 
		print ii,jj,'%.3f,%.3f'%(score_mat[ii,jj],cc)

np.save('%s/cc_mat.npy'%result_folder,cc_mat)
np.save('%s/score_mat.npy'%result_folder,score_mat)
'''
result_folder='/home/liucan/1tz9'
data=np.load('%s/cc_mat.npy'%result_folder)
num=data.shape[0]
print num

fig=plt.figure()
ax1=fig.add_subplot(111)
line1,=ax1.plot(range(num),np.mean(score_mat,axis=1),color='g')
ax1.set_xlabel('iteration numbers')
ax1.set_ylabel('average iq curve distance')
ax1.set_ylim(0,5)
ax2=ax1.twinx()
line2,=ax2.plot(range(num),np.mean(cc_mat,axis=1),color='b')
ax2.set_ylabel('average correlation coefficient')
plt.legend([line1,line2],['iq_distance-iter curve','cc-iter curve'],loc='center right')
plt.savefig('%s/result.png'%result_folder)
#plt.show()
plt.close()

plt.errorbar(range(num),np.mean(score_mat,axis=1),yerr=np.std(score_mat,axis=1),ecolor='r')
plt.xlabel('iteration numbers')
plt.ylabel('average iq curve distance')
plt.ylim(0,5)
plt.savefig('%s/iq_distance_curve.png'%result_folder)
#plt.show()
plt.close()

plt.errorbar(range(num),np.mean(cc_mat,axis=1),yerr=np.std(cc_mat,axis=1),ecolor='g')
plt.xlabel('iteration numbers')
plt.ylabel('average correlation coefficient')
plt.savefig('%s/correlation_cofficient.png'%result_folder)
#plt.show()
plt.close()
