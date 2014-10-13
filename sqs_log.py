import boto.sqs
import time

conn=boto.sqs.connect_to_region("us-east-1")
q=conn.get_queue("LearningMessages54")

f=open("54_s.log","a")
d={}

while True:
	rs=q.get_messages(attributes="All")
	if len(rs):
		for i in rs:
			mbody=i.get_body()
			mtime=int(i.attributes["SentTimestamp"])
			d[mtime]=mbody
			nos=mbody.split(",")
			if len(nos)==4:
				epoch=int(nos[0])
				train_p = float(nos[1])
				test_p = float(nos[3])
				print "%d %f %f"%(epoch, train_p, test_p)
				f.write("%d %f %f\n"%(epoch, train_p, test_p))
			else:
				print mbody
				f.write(mbody+"\n")
			f.flush()
			conn.delete_message_batch(q,rs)
	time.sleep(1)
