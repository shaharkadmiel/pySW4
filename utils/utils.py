"""
- utils.py -

Python module with general utilities.

By: Shahar Shani-Kadmiel, August 2012, kadmiel@post.bgu.ac.il

"""

def flatten(ndarray):
	"""Returns a flattened 1Darray from any multidimentional
	array. This function is recursive and may take a long time
	on large data"""
	for item in ndarray:
		try:
			for subitem in flatten(item):
				yield subitem
		except TypeError:
			yield item
