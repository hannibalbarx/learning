from boto.sqs.message import RawMessage
import boto.sqs
from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')
messages=parser.getboolean('config', 'messages')
queue = None

if messages:
	conn = boto.sqs.connect_to_region(
		"us-east-1")
	queue = conn.get_queue('LearningMessages')
	print 'connected to messaging queue'

def createQueue():
	conn = boto.sqs.connect_to_region("us-east-1")
	conn.get_all_queues()
	l=conn.get_all_queues()
	m=0
	for i in l:
		name = i.name
	a=int(l[1].name.split('LearningMessages')[1])+1

def sendMessage(m):
	print m
	if messages:
		raw_message = RawMessage()
		raw_message.set_body(m)
		queue.write(raw_message)
