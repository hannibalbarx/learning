import boto.sqs
import time

conn=boto.sqs.connect_to_region("us-east-1")
q=conn.get_queue("LearningMessages54")

f=open("54.log","wb")

while True:
	rs=q.get_messages(10,attributes="All")
	if len(rs):
		for i in rs:
			nos=i.get_body().split(",")
			if len(nos)==4:
				epoch=int(nos[0])
				train_p = float(nos[1])
				test_p = float(nos[3])
				print "%d %f %f"%(epoch, train_p, test_p)
				f.write("%d %f %f"%(epoch, train_p, test_p))
			else:
				print i.get_body()
				f.write(i.get_body())
			f.flush()
				
	time.sleep(1)