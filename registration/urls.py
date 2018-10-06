
from django.urls import path
from . import views



app_name = 'registration'
urlpatterns = [
	path('', views.index, name="index"), 
	path('home_page', views.home_page, name='home_page'),
	path('list_of_images', views.list_of_images, name='list_of_images'),
	path('test_print', views.test_print, name='test_print' ),
	path('upload', views.upload_file, name='upload_file'),
	
	path('images', views.images, name='images'),
	path('gallery', views.gallery, name='gallery'),
	
	path('register', views.register, name='register'),

	
	# path('grid_images', views.grid_images, name='grid_images')
	# path('download_images', views.download_images, name='download_images')

]
