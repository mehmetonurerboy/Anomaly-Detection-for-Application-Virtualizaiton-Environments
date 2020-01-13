from django.shortcuts import render
from django.http import HttpResponse
from .forms import SelectFile, ColumnSelection, RealTimeLR
from . import anomaly_detection
from . import uploadFile
from django.core.files.storage import FileSystemStorage
import time

from . import lr
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
    if (request.method == 'POST'):
        real_time_lr_form = RealTimeLR(request.POST)

        if real_time_lr_form.is_valid():
            cpu_percentage_value = real_time_lr_form.cleaned_data['cpu_percentage_input']
            ram_usage_value = real_time_lr_form.cleaned_data['ram_usage_input']
            io_usage_value = real_time_lr_form.cleaned_data['io_usage_input']
            network_usage_value = real_time_lr_form.cleaned_data['network_usage_input']



            spark = anomaly_detection.createSpark()
            df = anomaly_detection.readData('general_anomaly', spark)

            df_test = anomaly_detection.create_data_frame_from_list(spark,
                                                                    list=[[cpu_percentage_value],[ram_usage_value],[io_usage_value],[network_usage_value]],
                                                                    column_names=["cpu_percent","ram_usage","io_usage","network_usage"])

            lr_predictions=lr.calculateLogisticRegression(df,df_test)
            time_value = time.time()

            request.session['cpu_percentage'] = cpu_percentage_value
            request.session['ram_usage'] = ram_usage_value
            request.session['io_usage'] = io_usage_value
            request.session['network_usage'] = network_usage_value
            request.session['real-time_time_val'] = time_value
            request.session['anomaly_status'] = lr_predictions.iloc[0]["prediction"]

            print(cpu_percentage_value)
            print(ram_usage_value)
            print(io_usage_value)
            print(network_usage_value)

    else:
        real_time_lr_form = RealTimeLR()
    return render(request, 'input.html', {'form' : real_time_lr_form,
                                          'cpu_percentage' : request.session['cpu_percentage'],
                                          'ram_usage' : request.session['ram_usage'],
                                          'io_usage' : request.session['io_usage'],
                                          'network_usage' : request.session['network_usage'],
                                          'test_time' : request.session['real-time_time_val'],
                                          'anomaly_status' : request.session['anomaly_status']})


def data_input_form(request):
    if (request.method == 'POST'):
        form = SelectFile(request.POST, request.FILES)
        columnForm = ColumnSelection(request.POST)
        if columnForm.is_valid():

            selected_column_names = columnForm.cleaned_data['column_list_filter']
            time_column_name = columnForm.cleaned_data['time_field_filter']
            confidence_level = columnForm.cleaned_data['confidence_level_value']

            request.session['selected_columns'] = selected_column_names
            request.session['time_column_name'] = time_column_name
            request.session['confidence_level'] = confidence_level

            spark = anomaly_detection.createSpark()
            df = anomaly_detection.readData('statsData', spark)
            anomaly_values = anomaly_detection.obtain_anomaly_values(df, selected_column_names, confidence_level,
                                                                     time_column_name)

            anomaly_columns, normal_columns = anomaly_detection.obtain_anomaly_normal_values(df, anomaly_values)
            pandas_df = anomaly_detection.return_pandas_data_frame(df, anomaly_columns, selected_column_names)

            results_train_80_test_20 = anomaly_detection.data_test_for_80_perc_train_20_perc_tests_data(spark,
                                                                                                        pandas_df,
                                                                                                        selected_column_names,
                                                                                                        normal_columns,
                                                                                                        anomaly_values)
            results_10_fold = anomaly_detection.data_set_test_with_10_folds_cross_validation(spark,
                                                                                             selected_column_names,
                                                                                             pandas_df,
                                                                                             normal_columns,
                                                                                             anomaly_values)

            string = 'results_train_80_test_20'
            for indis in range(len(results_train_80_test_20)):
                request.session[string + str(indis)] = results_train_80_test_20[indis]

            string = 'results_10_fold'
            for indis in range(len(results_10_fold)):
                request.session[string + str(indis)] = results_10_fold[indis]

            request.session['cpu_percentage'] = " "
            request.session['ram_usage'] = " "
            request.session['io_usage'] = " "
            request.session['network_usage'] = " "
            request.session['real-time_time_val'] = " "
            request.session['anomaly_status'] = " "

            file_link = uploadFile.upload()
            request.session['csvLink'] = file_link

            return render(request, 'pca.html', {'csvLink': file_link})
    else:
        columnForm = ColumnSelection()

    return render(request, 'data_input.html', {'form': columnForm})


def handle_uploaded_file(f, fileName):
    with open(fileName, 'wb+') as destination:
        for chunk in f.chunks():
            destination.write(chunk)
