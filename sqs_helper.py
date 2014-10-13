from boto.sqs.message import RawMessage
import boto.sqs
from ConfigParser import SafeConfigParser

parser = SafeConfigParser()
parser.read('config.ini')

messages=parser.getboolean('config', 'messages')
if messages:
	conn = boto.sqs.connect_to_region("us-east-1")
	queue = conn.get_queue("LearningMessages")
	print 'connected to messaging queue'

def connected():
	return queue!=None
	
def send_message(m):
	print m
	if messages:
		raw_message = RawMessage()
		raw_message.set_body(m)
		queue.write(raw_message)

