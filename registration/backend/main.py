
from qmorph import *
from AutoRegister import *

"""
	Plan:

	
	*Create an UPLOADS directory where the uploaded images will go.
	*create a textfield in html
	*Create an upload button in an html file in templates 	
	*Create a funtion in views that will correspond to the html upload button
	and will upload a selected the selected image file into the UPLOADS directory
	*Create another html button in the templates called REGISTER
	*Have the registered 
"""


InputDir = 'uploads'
OutDir = 'registered'
flist = getFileNameList(InputDir, '*.xml')


doBatchRegistration(OutDir, warpMarkers= True, includeMarkers=True, applyMask=False, TemplateIndex=1)

# def register_images(request):
# 	# if image exists:
# 	# 	a = autoRegister(image)

# 	# return warp(a)
