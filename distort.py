import numpy
from pylearn2.utils import serial

def getpx_t(file):
	d = serial.load(file)

	d=d.reshape((50000,3,32,32))
	dtp=d.copy()
	dtn=d.copy()
	for i in range(0,50000):
		for j in range(0,3):
			#d[i,j]=numpy.fliplr(d[i,j])
			dtp[i,j]=numpy.dot(d[i,j], numpy.eye(32,k=5))
			dtn[i,j]=numpy.dot(d[i,j], numpy.eye(32,k=-5))
	dtp=dtp.reshape((50000,3072))
	dtn=dtn.reshape((50000,3072))
	
	serial.save("tp_"+file,dtp)
	serial.save("tn_"+file,dtn)

def getpx_r(file):
	d = serial.load(file)

	d=d.reshape((50000,3,32,32))
	for i in range(0,50000):
		for j in range(0,3):
			d[i,j]=numpy.fliplr(d[i,j])
	d=d.reshape((50000,3072))
	
	serial.save("r_"+file,d)

def shower(m):
    im = Image.new("RGB", m[0].shape, "white")
    for i in range(0,m[0].shape[0]):
	    for j in range(0,m[0].shape[1]):
		    im.putpixel((i,j),(m[0,j,i],m[1,j,i],m[2,j,i]))
    im.show()

if __name__ == '__main__':
	getpx_r('/home/ec2-user/permalearn/cifar10/pylearn2_gcn_whitened/train.npy')
	getpx_t('/home/ec2-user/permalearn/cifar10/pylearn2_gcn_whitened/train.npy')
