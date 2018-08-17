
from django.urls import path
from . import views



app_name = 'registration'
urlpatterns = [
	path('', views.index, name="index"),
	# path('', views.home_page, name="home_page"),
	path('pr', views.pr, name='pr' ),
	path('upload', views.upload_file, name='upload_file'),
	path('home', views.home, name='home'),

	path('images', views.images, name='images'),
	path('register', views.register, name='register'),
	path('gallery', views.gallery, name='gallery'),
	path('home_page', views.home_page, name='home_page'),
	# path('grid_images', views.grid_images, name='grid_images')
	# path('download_images', views.download_images, name='download_images')

]
