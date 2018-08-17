from django import forms

from registration.models import Document


class DocumentForm(forms.ModelForm):
    class Meta:
        model = Document
        fields = ('description', 'document', )
        # app_label = 'registration'



# class UploadFileForm(forms.Form):
# 	title = forms.CharField(max_length=50)
# 	file = forms.FileField()


# class ModelFormWithFileField(models.Model):
#     # file will be uploaded to MEDIA_ROOT/uploads
#     upload = models.FileField(upload_to='uploads/')
#     # or...
#     # file will be saved to MEDIA_ROOT/uploads/2015/01/30
#     # upload = models.FileField(upload_to='uploads/%Y/%m/%d/')


