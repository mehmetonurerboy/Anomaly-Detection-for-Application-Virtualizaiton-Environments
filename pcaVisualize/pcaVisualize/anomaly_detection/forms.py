from django import forms

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
                                                   label="Select the dimensions that are numeric and will be used on PCA-based anomaly detection method",
                                                   widget=forms.CheckboxSelectMultiple(attrs={'class' : 'form-control'}),
                                                   choices=column_enum)

    time_field_filter = forms.ChoiceField(label='Select the dimension that contain the time epoch value',
                                          choices=column_enum,
                                          widget=forms.Select(attrs={'class' : 'form-control'}))

    confidence_level_value = forms.FloatField(required=True,
                                              max_value=1,
                                              min_value=0)
