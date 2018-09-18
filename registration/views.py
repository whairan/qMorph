from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect, Http404
from django.template import loader, context

from .models import Document, Image
from .forms import DocumentForm

from django.conf import settings
from django.http import HttpResponse


from registration.backend import AutoRegister
import os

def pr(request):
	print("hello")
	return HttpResponse("anything")


def home(request):
    uploads = Document.objects.all()
    return render(request, 'registration/home.html', { 'uploads': uploads })

def home_page(request):
	return render(request, 'registration/home_page.html')


def index(request):
	template = loader.get_template('registration/menu.html')
	context = {"a":"b"}
	return HttpResponse(template.render(context, request))

def register_image(request):
	from registration.warp import warp
	return render(request, 'registration/warp.html', {"warp":warp})

def upload_file(request):
	"""This function allows the user to upload the image files"""
	if request.method == 'POST':
		form = DocumentForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			return HttpResponseRedirect('upload')
	else:
		form = DocumentForm()
	return render(request, 'registration/upload.html', {'form': form})


# def download(request, path):
# 	"""This function allows the user to download the registered file"""
# 	file_path = os.path.join(settings.MEDIA_ROOT, path)
# 	if os.path.exists(file_path):
# 	    with open(file_path, 'rb') as fh:
# 	    	content_type = magic.from_buffer(image_buffer, mime=True)
# 	    	response = HttpResponse(fh.read(), content_type=content_type)
# 	    	response['Content-Disposition'] = 'inline; filename=' + os.path.basename(file_path)
# 	    	return response
# 	raise Http404

def images(request):
  images = Image.objects.all()
  return render(request, "registration/display_image.html", {'images': images})

def register(request):
	"""This function takes the uploaded files and registers them"""
	print("It works, yay!")
	return HttpResponse("fix the registration function in Views")


def gallery(request):
    path = "/Users/wall-E/Desktop/imgReg/uploads/"
    img_list = os.listdir(path)
    return render(request, 'registration/display_image.html', {'imgs': img_list})

# def read_image(path = None, stream = None, url= None):
# 	if path is not None:
# 		image = cv2.imread(path)
# 	else:
# 		if url is not None:
# 			response = urllib.request.urlopen(url)
# 			data_temp = response.read()
# 		elif stream is not None:
# 			data_temp = stream.read()
# 		image = np.asarray(bytearray(data_temp), dtype='unit8')
# 		image = cv2.imdecode(image, cv2.IMREAD_COLOR)

# 	return image


