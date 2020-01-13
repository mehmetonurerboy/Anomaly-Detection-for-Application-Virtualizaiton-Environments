from django import forms
from django.utils.safestring import mark_safe

class SelectFile(forms.Form):
    file = forms.FileField(label='Select File ')


class ColumnSelection(forms.Form):
    column_enum = {
        ("ktime", "Ktime"),
        ("container", "Container"),
        ("node", "Node Name"),
        ("io_usage", "IO Usage"),
        ("time", "Time"),
        ("pids", "PIDS"),
        ("ram_limit", "RAM Limit"),
        ("cpu_percent", "CPU Percent"),
        ("io_limit", "IO Limit"),
        ("network_limit", "Network Limit"),
        ("network_usage", "Network Usage"),
        ("ram_usage", "RAM Usage"),
        ("customer_id", "Customer ID"),
        ("application_id","Application ID"),
    }

    column_list_filter = forms.MultipleChoiceField(required=True,
                                                   label=mark_safe('<font size="3"> Select the dimensions that are numeric and will be used on PCA-based anomaly detection method </font> <br>'),
                                                   widget=forms.CheckboxSelectMultiple(attrs={'class' : 'form-control'}),
                                                   choices=column_enum)

    time_field_filter = forms.ChoiceField(label=mark_safe('<font size="3"> Select the dimension that contain the time epoch value </font> <br>'),
                                          required=True,
                                          choices=column_enum,
                                          widget=forms.Select(attrs={'class' : 'form-control'}))

    confidence_level_value = forms.FloatField(required=True,
                                              label=mark_safe('<br><br><br> <font size="3"> Confidence Level </font> <br>'),
                                              max_value=1,
                                              min_value=0)


class RealTimeLR(forms.Form):
    cpu_percentage_input = forms.FloatField(label=mark_safe('<font size="2"> CPU Percentage </font> <br>'),
                                            required=True,
                                            max_value=150,
                                            min_value=0)

    ram_usage_input = forms.FloatField(label=mark_safe('<font size="2"> RAM Usage </font> <br>'),
                                       required=True)

    io_usage_input = forms.FloatField(label=mark_safe('<font size="2"> IO Usage </font> <br>'),
                                      required=True)

    network_usage_input = forms.FloatField(label=mark_safe('<font size="2"> Network Usage </font> <br>'),
                                           required=True)
