import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Simulasi Fuzzy",layout='wide')

# code 
# The distance set. "start" and "end" represents the x-axis and each variable is represented by [x0, x1, x2].
distance_set = {
        "start": 0,
        "end": 10,
        "VerySmall": [0, 1, 2.5],
        "Small": [1.5, 3, 4.5],
        "Perfect": [3.5, 5, 6.5],
        "Big": [5.5, 7, 8.5],
        "VeryBig": [7.5, 9, 10]
    }

# The delta set. "start" and "end" represents the x-axis and each variable is represented by [x0, x1, x2].
delta_set = {
        "start": -5,
        "end": 5,
        "ShrinkingFast": [-5, -4, -2.5],
        "Shrinking": [-3.5, -2, -0.5],
        "Stable": [-1.5, 0, 1.5],
        "Growing": [0.5, 2, 3.5],
        "GrowingFast": [2.5, 4, 5]
    }

# The action set. "start" and "end" represents the x-axis and each variable is represented by [x0, x1, x2].
action_set = {
        "start": -10,
        "end": 10,
        "BrakeHard": [-10, -8, -5],
        "SlowDown": [-7, -4, -1],
        "None": [-3, 0, 3],
        "SpeedUp": [1, 4, 7],
        "FloorIt": [5, 8, 10]
    }

# Return y value from a triangle shape.
def triangle(position, x0, x1, x2, clip):
    value = 0.0

    if x0 <= position <= x1:
        value = (position - x0) / (x1 - x0)
    elif x1 <= position <= x2:
        value = (x2 - position) / (x1 - x0)

    if value > clip:
        value = clip
    # st.header("triangle")
    # st.write(value)
    return value

# Return y value from a grade shape.
def grade(position, x0, x1, clip):
    if position >= x1:
        value = 1.0
    elif position <= x0:
        value = 0.0
    else:
        value = (position - x0) / (x1 - x0)

    if value > clip:
        value = clip

    # st.header("grade")
    # st.write(value)
    return value

# Return y value from a reverse grade shape.
def reverse_grade(position, x0, x1, clip):
    if position <= x0:
        value = 1.0
    elif position >= x1:
        value = 0.0
    else:
        value = (x1 - position) / (x1 - x0)

    if value > clip:
        value = clip

    # st.header("reverse_grade")
    # st.write(value)
    return value

# Step 1. Fuzzification of the input variables
def fuzzification(variables, x, clip):
    fuzzy_degrees = []

    # Loops through every name and value in the variable.
    for set_name, set_value in variables.items():

        # If name is not start or end - meaning current element is a set.
        if set_name != "start" and set_name != "end":

            # If shape is reverse grade.
            if set_value[0] == variables["start"]:
                y = reverse_grade(position=x, x0=set_value[1], x1=set_value[2], clip=clip)
                fuzzy_degrees.append([set_name, y])

            # If shape is grade.
            elif set_value[2] == variables["end"]:
                y = grade(position=x, x0=set_value[0], x1=set_value[1], clip=clip)
                fuzzy_degrees.append([set_name, y])

            # Shape is triangle.
            else:
                y = triangle(position=x, x0=set_value[0], x1=set_value[1], x2=set_value[2], clip=clip)
                fuzzy_degrees.append([set_name, y])

    # Return list
    # st.header("Fuzzification")
    # st.write(fuzzy_degrees)
    return fuzzy_degrees

# Step 2. Rule evaluation (inference)
def rule_evaluation(set_1, set_2):
    action_outputs = []

    # Delta variables used in rule.
    not_growing = 0.0
    not_growing_fast = 0.0

    # Set delta variables.
    for delta in set_2:
        if delta[0] == "Growing":
            not_growing = 1 - delta[1]
        elif delta[0] == "GrowingFast":
            not_growing_fast = 1 - delta[1]

    # Set action outcomes according to rules.
    for distance in set_1:
        for delta in set_2:

            # IF distance is Small AND delta is Growing THEN action is None
            if distance[0] == "Small" and delta[0] == "Growing":
                action_outputs.append(["None", min(distance[1], delta[1])])

            # IF distance is Small AND delta is Stable THEN action is SlowDown
            elif distance[0] == "Small" and delta[0] == "Stable":
                action_outputs.append(["SlowDown", min(distance[1], delta[1])])

            # IF distance is Perfect AND delta is Growing THEN action is SpeedUp
            elif distance[0] == "Perfect" and delta[0] == "Growing":
                action_outputs.append(["SpeedUp", min(distance[1], delta[1])])

            # IF distance is VeryBig AND (delta is NOT Growing OR delta is NOT GrowingFast) THEN action is FloorIt
            elif distance[0] == "VeryBig" and (delta[0] != "Growing" and delta[0] != "GrowingFast"):
                element = ["FloorIt", min(distance[1], max(not_growing, not_growing_fast))]
                if element not in action_outputs:
                    action_outputs.append(element)

            # IF distance is VerySmall THEN action is BrakeHard
            elif distance[0] == "VerySmall":
                element = ["BrakeHard", distance[1]]
                if element not in action_outputs:
                    action_outputs.append(element)

    # action_outputs = [['BrakeHard', 0.0], ['SlowDown', 0.25], ['None', 0.4], ['SpeedUp', 0.1], ['FloorIt', 0.0]]
    # st.header("Inference")
    # st.write(action_outputs)
    return action_outputs

# Step 3. Aggregation of the rule outputs (composition)
def aggregation(clipped_list):
    # Changes format of clipped_List to [name, value, start, end] by appending x value for start and end.
    # This is done for summation purposes in step 4.

    # Set variables to be used in for loop
    x_start = action_set["start"]
    current_max_name = clipped_list[0][0]

    # Loop through every x value in the action set
    for x in range(action_set["start"], action_set["end"] + 1):

        # Get max set of fuzzification
        max_value_for_x = max(fuzzification(action_set, x, 1), key=lambda m: m[1])
        curr_name = max_value_for_x[0]

        # If current maximum name is not curr name
        if current_max_name != curr_name:

            # Loop through clipped list
            for i in clipped_list:

                # Add x value for start and end
                if i[0] == current_max_name:
                    i.append(x_start)
                    i.append(x - 1)

            # Update variables
            current_max_name = curr_name
            x_start = x

        # If last x value in action set
        elif x == action_set["end"]:

            # Loop through clipped list
            for i in clipped_list:

                # Add x value for start and end
                if i[0] == current_max_name:
                    i.append(x_start)
                    i.append(x)

    return clipped_list

# Step 4. Defuzzification
def defuzzification(aggregated_set):
    numerator_sum = 0.0
    denominator_sum = 0.0

    # Loop through every item in the aggregated set.
    for i in aggregated_set:
        value = i[1]
        start = i[2]
        end = i[3]

        # Represents each term between the addition signs (+).
        item_sum = 0.0

        # Loops through every x value, for each set.
        for x in range(start, end + 1):
            # Adds each x value to item_sum variable
            item_sum += x

            # Adds action value to the denominator
            denominator_sum += value

        # Finishes a term in the numerator sum by adding the x values with the action value.
        numerator_sum += item_sum * value

    # Returns the center of gravity (COG)
    # st.header("Defuzzification")
    # st.write(numerator_sum / denominator_sum)
    try:
        return numerator_sum / denominator_sum
    except ZeroDivisionError:
        return 0

    # return numerator_sum / denominator_sum

def start_fuzzy(distance_input, delta_input):
    # Step 1 - Fuzzification
    col7, col8 = st.beta_columns([1,1])
    distance_values = fuzzification(distance_set, distance_input, 1)
    with col7:
        data = np.array(distance_values)
        df = pd.DataFrame({'Data': data[:, 0], 'Nilai': data[:, 1]})
        st.subheader("Fuzzification - Distance")
        st.dataframe(df)
    
    with col8:
        delta_values = fuzzification(delta_set, delta_input, 1)
        data = np.array(delta_values)
        df = pd.DataFrame({'Data': data[:, 0], 'Nilai': data[:, 1]})
        st.subheader("Fuzzification - Delta")
        st.dataframe(df)
        # st.write(distance_values)
        # st.write(delta_values)

    # Step 2 - Rule evaluation (inference)
    actions = rule_evaluation(distance_values, delta_values)
    st.header("Rule evaluation / Operasi Fuzzy")
    data = np.array(actions)
    df = pd.DataFrame({ 'Nilai': data[:, 1], 'Output': data[:, 0]}, index = ['R1','R2','R3','R4','R5'])
    st.dataframe(df)
    # st.write(actions)

    # Step 3 - Aggregation
    aggregation_output = aggregation(actions)
    st.header("Aggregation / Komposisi Aturan")
    data = np.array(aggregation_output)
    df = pd.DataFrame(data)
    df_t = df.T
    df_t.columns = df_t.iloc[0]
    df_new = df_t.drop(df_t.index[0])
    st.dataframe(df_new.rename(index={1:'Nilai',2:'Min',3:'Max' }))
    # st.write(aggregation_output)

    # Step 4 - Defuzzification
    output = defuzzification(aggregation_output)
    print("COG output:", output)
    st.header("Defuzzification")
    st.write("Centorid output:", output)

    fuzzified_output = fuzzification(action_set, output, 1)
    resulting_action = [fuzzified_output[0][0], fuzzified_output[0][1]]
    for item in fuzzified_output:
        if item[1] > resulting_action[1]:
            resulting_action = [item[0], item[1]]

    print("The robot will take following action:", resulting_action)
    st.write("Mobil Akan Melakukan aksi :", resulting_action)


st.title('Simulasi Fuzzy pada kecepatan mobil')
st.markdown("""
Mobil secara otomatis mengikuti objek yang bergerak. Hal penting yang harus diperhatikan adalah menjaga jarak
dengan mengontrol percepatan dalam kaitannya dengan pergerakan benda. 
Untuk pengendaraan yang baik, fleksibel dan kontinyu, pengendalian mobil dilakukan dengan
menggunakan penalaran fuzzy. 
Ada dua variabel dalam robot yang menentukan respons: 
* jarak (distance) --> jarak ke mobil di depan delta 
* delta --> perubahan jarak per. satuan waktu
""")
col1, col2 = st.beta_columns([4,2])

with col1:
    st.subheader("Rules")
    st.image('./rules.png', use_column_width=True)

with col2:
    st.subheader("Output")
    st.image('./action.png', use_column_width=True)

col3, col4 = st.beta_columns([2,2])

with col3:
    dataset = {'Variabel': ['Distance','Delta','Action'],
            'Type': ['input','input','output'],
            'Min-Kondisi':['very small','shrinking fast', 'brake hard'],
            'Min-value':['0','-5', '-10'],
            'max-Kondisi':['very big','GrowingFast', 'FloorIt'],
            'max-value':['10','5', '10'],
            }

    df = pd.DataFrame(dataset, columns = ['Variabel', 'Type', 'Min-Kondisi', 'Min-value', 'max-Kondisi', 'max-value'])
    st.subheader("Inisiasi")
    st.dataframe(df)

with col4:
    Rules = {'Param-1': ['distance','distance','distance','distance','distance',],
            'Des-1':['small','small', 'perfect', 'very big', 'very small'],
            'Param-2':['delta','delta','delta','delta',''],
            'Des-2':['Growing','stable', 'Growing','NOT Growing / NOT Growing Fast',''],
            'Param-3':['action','action','action','action','action'],
            'Des-3':['None','Slow Down', 'Speed Up', 'Floor It', 'Brake hard'],
            }

    df = pd.DataFrame(Rules, columns = ['Param-1', 'Des-1', 'Param-2', 'Des-2', 'Param-3','Des-3'], index = ['R1','R2','R3','R4','R5'])
    st.subheader("Rules")
    st.dataframe(df)

col5, col6 = st.beta_columns([2,2])

with col5:
    distance = int(st.slider('Distance', min_value=0, max_value=10))
    delta = int(st.slider('Delta', min_value=-5, max_value=5))
    start = st.button('Mulai')

with col6:
    st.header('Input')
    st.write('Distance = ', distance)
    st.write('Delta = ', delta)

# st.header('Proses Fuzzy')

if start:
    start_fuzzy(distance, delta)

