from boto.sqs.message import RawMessage
import boto.sqs
from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')

messages=parser.getboolean('config', 'messages')
if messages:
	conn = boto.sqs.connect_to_region("us-east-1")
queue = None

def connected():
	return queue!=None
	
def connect_queue():
	if messages:
		global queue
		queue_number, queue = get_last_queue()
		print 'connected to messaging queue'
		
def create_queue():
	if messages:
		global queue
		queue_number, queue_old = get_last_queue()
		queue = conn.create_queue('LearningMessages'+str(queue_number+1))
		print 'created messaging queue '+queue.name

def send_message(m):
	print m
	if messages:
		raw_message = RawMessage()
		raw_message.set_body(m)
		queue.write(raw_message)

def get_last_queue():
	l=conn.get_all_queues()
	last_queue_number=0
	last_queue=None
	for i in l:
		queue_number=int(i.name.split('LearningMessages')[1])
		if queue_number>last_queue_number:
			last_queue_number=queue_number
			last_queue=i
	return last_queue_number, last_queue

	