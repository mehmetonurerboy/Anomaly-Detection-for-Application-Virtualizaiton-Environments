from django.shortcuts import render
from django.http import HttpResponse
from .forms import SelectFile, ColumnSelection

# Create your views here.
def show_pca(request):
    return render(request, 'pca.html', {})

def show_lr(request):
    return render(request, 'lr.html', {})

def show_input(request):
    return render(request, 'input.html', {})

def data_input_form(request):
    if (request.method == 'POST') and (request.FILES):
        form = SelectFile()
        file = SelectFile(request.POST, request.FILES)
        if file.is_valid():
            file = request.FILES['file']
            handle_uploaded_file(file, 'dataset.csv')
            columnForm = ColumnSelection()

            selected_column_name = form.cleaned_data['column_list_filter']
            time_column_name = form.cleaned_data['time_field_filter']
            confidence_level = form.cleaned_data['confidence_level_value']

            return render(request, 'form.html', {'file_selection' : form , 'form': columnForm})
            print(selected_column_name)
            print(time_column_name)
        else:
            print("some problem ocurred")
    else:
        form = ColumnSelection()
    return render(request, 'data_input.html', {'form' : form})

def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
