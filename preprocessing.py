import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_data(file_path):
    col_names = ['ID', 'Age', 'PD', 'TD', 'Visual_Condition', 'Neutral_Corrosion']
    data = pd.read_excel(file_path, names=col_names)
    data.set_index(['ID', 'Age'],inplace=True)
    data.sort_index(inplace=True)
    data.reset_index(inplace=True)
    data.set_index('ID',inplace=True)

    return data

def scale_data(data):
    index = data.index
    scaler_X = MinMaxScaler()
    scaler_X.fit(data) #fit to data
    scaled_data = scaler_X.transform(data)
    scaled_data = pd.DataFrame(scaled_data)
    scaled_data.insert(0,'ID',index)
    col_names = ['ID', 'Age', 'PD', 'TD', 'Visual_Condition', 'Neutral_Corrosion']
    scaled_data.columns = col_names

    return scaled_data

def HI_computation(data, scaled_data):
    # weights
    w_availability_age = 3
    w_availability_PD = 1
    w_availability_TD = 1
    w_availability_neutral_corrosion = 2
    w_availability_visual_condition = 2
    w_operator_age = 3
    w_operator_PD = 2
    w_operator_TD = 2
    w_operator_neutral_corrosion = 1
    w_operator_visual_condition = 1
    w_instrumentation_age = 3
    w_instrumentation_PD = 1
    w_instrumentation_TD = 1
    w_instrumentation_neutral_corrosion = 3
    w_instrumentation_visual_condition = 3
    w_age = 1
    w_PD = 2
    w_TD = 2
    w_neutral_corrosion = 3
    w_visual_condition = 3
    
    data.reset_index(inplace = True)
    
    scaled_data['HI'] = scaled_data['Age']*(w_age+w_operator_age+w_availability_age+w_instrumentation_age) + scaled_data['PD']*(w_PD+w_operator_PD+w_availability_PD+w_instrumentation_PD) + scaled_data['TD']*(w_TD+w_operator_TD+w_availability_TD+w_instrumentation_TD) + scaled_data['Visual_Condition']*(w_visual_condition+w_operator_visual_condition+w_availability_visual_condition+w_instrumentation_visual_condition) + scaled_data['Neutral_Corrosion']*(w_neutral_corrosion+w_operator_neutral_corrosion+w_availability_neutral_corrosion+w_instrumentation_neutral_corrosion)
    data['HI'] = round(scaled_data['HI'],0)
    
    return data

def one_hot_encoding(data):
    data['ID'] = data['ID'].astype('category')
    data['Visual_Condition'] = data['Visual_Condition'].astype('category')
    data = pd.get_dummies(data, columns=['ID', 'Visual_Condition'])
    
    return data


