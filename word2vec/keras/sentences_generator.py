import codecs

class Sentences(object):
	"""docstring for Sentences"""
	def __init__(self, filepath):
		self.filepath = filepath
	
	def __iter__(self):
		with codecs.open(self.filepath, "r", "utf-8") as rf:
			for line in rf:
				if line.strip():
					yield line.strip().lower()
