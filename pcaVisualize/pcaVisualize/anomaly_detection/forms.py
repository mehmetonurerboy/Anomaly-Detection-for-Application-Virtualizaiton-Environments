from django import forms

class SelectFile(forms.Form):
    file = forms.FileField(label='Select File ')


class ColumnSelection(forms.Form):
    column_enum = {
        ("ktime", "Ktime Value"),
        ("container", "Container Value"),
        ("node", "Node Name"),
        ("io_usage", "IO Usage"),
        ("time", "Time Value"),
        ("pids", "PIDS Value"),
        ("ram_limit", "RAM Limit Value"),
        ("cpu_percent", "CPU Percent Value"),
        ("io_limit", "IO Limit Value"),
        ("network_limit", "Network Limit Value"),
        ("network_usage", "Network Usage Value"),
        ("ram_usage", "RAM Usage Value"),
        ("customer_id", "Customer ID Value"),
        ("application_id","Application ID Value"),
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
