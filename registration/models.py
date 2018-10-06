

from django.db import models

# Create your models here.


class Document(models.Model):
    description = models.CharField(max_length=255, blank=True)
    document = models.FileField(upload_to='uploads/')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    # app_label = 'registration'

    #if description == ideal:
    #then download to 

class Image(models.Model):
    img = models.ImageField(upload_to="uploads/Ideal",null=True, blank=True)

class Posting(models.Model):
    title = models.CharField(max_length=150)
    images = models.ForeignKey(Image, null=True,on_delete=models.CASCADE)



# class upload_img_file(models.Model):
#     # file will be uploaded to MEDIA_ROOT/uploads
#     upload = models.FileField(upload_to='registration/uploads/')
#     # or...
#     # file will be saved to MEDIA_ROOT/uploads/2015/01/30
#     # upload = models.FileField(upload_to='uploads/%Y/%m/%d/')


# class ImageA(models.Model):
#         title = models.CharField(max_length=0)
#         category = models.CharField(max_length=50,choices=CATEGORY_CHOICES)
#         thumbnail = models.ImageField(upload_to = 'uploaded_images/')


# class TissueImage(models.Model):
#     vegetable = models.ForeignKey(ImageA, default=None, related_name='images')
#     image = models.ImageField(upload_to='images/vegetable',
#                              verbose_name='image',)