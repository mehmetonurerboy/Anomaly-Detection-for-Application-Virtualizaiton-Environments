from django.shortcuts import render
from django.http import HttpResponse
from .forms import SelectFile, ColumnSelection
from . import anomaly_detection
from . import uploadFile
import time

file_link = ""
results_train_80_test_20 = []
results_10_fold = []

# Create your views here.
def show_pca(request):
    return render(request, 'pca.html', {'csvLink' : request.session['csvLink']})

def show_lr(request):
    return render(request, 'lr.html', {
        'result_80_train_20_test_table_val_1': request.session['results_train_80_test_200'],
        'result_80_train_20_test_table_val_2': request.session['results_train_80_test_201'],
        'result_80_train_20_test_table_val_3': request.session['results_train_80_test_202'],
        'result_80_train_20_test_table_val_4': request.session['results_train_80_test_203'],
        'result_80_train_20_test_accuracy_rate': request.session['results_train_80_test_204'],
        'result_80_train_20_test_f_score': request.session['results_train_80_test_205'],
        'result_10_fold_table_val_1': request.session['results_10_fold0'],
        'result_10_fold_table_val_2': request.session['results_10_fold1'],
        'result_10_fold_table_val_3': request.session['results_10_fold2'],
        'result_10_fold_table_val_4': request.session['results_10_fold3'],
        'result_10_fold_accuracy_rate': request.session['results_10_fold4'],
        'result_10_fold_f_score': request.session['results_10_fold5'],
    })

def show_input(request):
    return render(request, 'input.html', {})

def data_input_form(request):
    if (request.method == 'POST'):

        form = ColumnSelection(request.POST)
        #handle_uploaded_file('dataset.csv')
        columnForm = ColumnSelection(request.POST)
        if columnForm.is_valid():
            selected_column_names = columnForm.cleaned_data['column_list_filter']
            time_column_name = columnForm.cleaned_data['time_field_filter']
            confidence_level = columnForm.cleaned_data['confidence_level_value']

            spark = anomaly_detection.createSpark()
            df = anomaly_detection.readData('statsData', spark)
            anomaly_values = anomaly_detection.obtain_anomaly_values(df, selected_column_names, confidence_level,
                                                                     time_column_name)

            anomaly_columns, normal_columns = anomaly_detection.obtain_anomaly_normal_values(df,anomaly_values)
            pandas_df = anomaly_detection.return_pandas_data_frame(df,anomaly_columns,selected_column_names)
            results_train_80_test_20 = anomaly_detection.data_test_for_80_perc_train_20_perc_tests_data(spark,pandas_df,selected_column_names,normal_columns,anomaly_values)
            results_10_fold = anomaly_detection.data_set_test_with_10_folds_cross_validation(spark,selected_column_names,pandas_df,normal_columns,anomaly_values)

            string = 'results_train_80_test_20'
            for indis in range(len(results_train_80_test_20)):
                request.session[string + str(indis)] = results_train_80_test_20[indis]

            string = 'results_10_fold'
            for indis in range(len(results_10_fold)):
                request.session[string + str(indis)] = results_10_fold[indis]

            file_link = uploadFile.upload()
            request.session['csvLink'] = file_link
            return render(request, 'pca.html', {'csvLink' : file_link})

    else:
        form = ColumnSelection()
    return render(request, 'data_input.html', {'form' : form})

def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
