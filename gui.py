import tkinter
from tkinter import *
import numpy as np
import pickle

from tensorflow.keras.models import model_from_json
import os

os.chdir('files/')
#loading standard scaler
sc = pickle.load(open('standard_scaler.sav', 'rb'))

# Loading ann
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
ann = model_from_json(loaded_model_json)
# load weights into new model
ann.load_weights("weights_best.hdf5")
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Loading tree
decision_tree = pickle.load(open("best_tree.sav", 'rb'))

# Loading random forest
random_forest = pickle.load(open("best_forest.sav", 'rb'))

root = Tk()
frame = tkinter.Frame(root)
frame.grid(row=0, column=0)

attributes = ["HbA1c", "FBS", "Cholesterol", "LDL", "BMI", "Age", "sBP", "HDL", "TG", "Sex", "Depression",
              "HTN", "OA", "COPD"]

udm = ['mmol/L', 'mmol/L', 'mmol/L', 'mmol/L', 'kg/m^2', 'years', 'mmHg', 'mmol/L', 'mmol/L']
labels = []
udm_labels = []
error_labels = []
entries = []
buttons = []
graphicLabel = Label(frame)
graphicLabel.config(text="   ")
graphicLabel.grid(row = 5, column = 6)
instructionLabel = Label(frame)
instructionLabel.config(text="Insert your data to know if\nyou are risking to develop\ntype 2 diabetes mellitus")
instructionLabel.grid(row = 5, column = 7)
riskLabel = Label(frame)
riskLabel.config(text = ' ')
riskLabel.grid(row = 6, column = 7)
prescriptionLabel = Label(frame)
prescriptionLabel.config(text=' ')
prescriptionLabel.grid(row = 7, column = 7)

ordered_attributes = ['Age', 'sBP', 'BMI', 'LDL', 'HDL', 'HbA1c', 'TG', 'FBS', 'Cholesterol',
                      'Depression', 'HTN', 'OA', 'COPD', 'Sex']
patient_data = np.zeros(shape=(1, len(ordered_attributes)))
numerical_values_inserted = False
binary_values_inserted = False
sex_inserted = False
depression_inserted = False
HTN_inserted = False
OA_inserted = False
COPD_inserted = False

def reset():
    global numerical_values_inserted
    global binary_values_inserted
    global sex_inserted
    global depression_inserted
    global HTN_inserted
    global OA_inserted
    global COPD_inserted
    global patient_data
    global binary_labels
    patient_data = np.zeros(shape=(1, len(ordered_attributes)))
    numerical_values_inserted = False
    binary_values_inserted = False
    sex_inserted = False
    depression_inserted = False
    HTN_inserted = False
    OA_inserted = False
    COPD_inserted = False
    for i in range(len(buttons)):
        buttons[i].config(relief = RAISED)

    for i in range(len(error_labels)):
        error_labels[i].config(text = ' ')

    for i in range(len(udm_labels)):
        udm_labels[i].config(text = udm[i])

    for i in range(len(entries)):
        entries[i].delete(0, 'end')

    instructionLabel.config(text = "Insert your data to know if\nyou are risking to develop\ntype 2 diabetes mellitus")
    prescriptionLabel.config(text = ' ')
    riskLabel.config(text = ' ', fg = 'black')

def all_binary_values_inserted():
    return sex_inserted*depression_inserted*HTN_inserted*OA_inserted*COPD_inserted

def print_prescription():
    # ordered_attributes = ['Age', 'sBP', 'BMI', 'LDL', 'HDL', 'HbA1c', 'TG', 'FBS', 'Cholesterol',
                          # 'Depression', 'HTN', 'OA', 'COPD', 'Sex']
    global ordered_attributes
    global patient_data

    intro = 'The decision tree has classified you to be at risk of developing T2DM for the following reason:\n'
    prescription = ''

    Age = patient_data[0, ordered_attributes.index('Age')]
    sBP = patient_data[0, ordered_attributes.index('sBP')]
    BMI = patient_data[0, ordered_attributes.index('BMI')]
    LDL = patient_data[0, ordered_attributes.index('LDL')]
    HDL = patient_data[0, ordered_attributes.index('HDL')]
    HbA1c = patient_data[0, ordered_attributes.index('HbA1c')]
    TG = patient_data[0, ordered_attributes.index('TG')]
    FBS = patient_data[0, ordered_attributes.index('FBS')]
    Cholesterol = patient_data[0, ordered_attributes.index('Cholesterol')]
    Depression = patient_data[0, ordered_attributes.index('Depression')]
    HTN = patient_data[0, ordered_attributes.index('HTN')]
    OA = patient_data[0, ordered_attributes.index('OA')]
    COPD = patient_data[0, ordered_attributes.index('COPD')]
    Sex = patient_data[0, ordered_attributes.index('Sex')]

    if HbA1c <= 6.14:
        if FBS > 6.95:
            prescription = 'FBS > 6.95'
        else:
            if HbA1c > 5.94 and LDL <= 2.61:
                prescription = 'LDL <= 2.61'
    else:
        if HbA1c > 6.615:
            prescription = 'HbA1c > 6.615'
        else:
            if FBS > 6.35:
                prescription = 'FBS > 6.35'
            else:
                if LDL <= 2.575:
                    prescription = 'LDL <= 2.575'
                else:
                    if FBS >= 5.55:
                        prescription = 'FBS >= 5.55'

    prescriptionLabel.config(text=intro+prescription)


def printOutcome(result, votes):
    global instructionLabel
    tree_out = "Not at risk"
    forest_out = "Not at risk"
    ann_out = "Not at risk"
    result_out = "not"
    colour = 'blue'
    if votes[0] == 1:
        tree_out = "At risk"
    if votes[1] == 1:
        forest_out = "At risk"
    if votes[2] == 1:
        ann_out = "At risk"
    if result == 1:
        result_out = ""
        colour = 'red'
    instructionLabel.config(text="The prediction for the three algorithms are:\nDT = " + tree_out + "\nRF = " + forest_out +
                        "\nANN = " + ann_out + "\nSince our algorithms has the best performance when two or more of the"
                                 + "\nthree algorithms agree with each other you result ")



    riskLabel.config(text= result_out + " to be at risk of developing DMT2", fg = colour)

    if result == 1 and votes[0] == 1:
        print_prescription()


def confirmation():
    global attributes
    global entries
    global ordered_attributes
    global patient_data
    instructionLabel.config(text =' ')
    prescriptionLabel.config(text = ' ')
    riskLabel.config(text=' ', fg='black')
    all_valid = True
    for i in range(len(attributes[:9])):
        value = entries[i].get()
        try:
            value = float(value)
            idx = ordered_attributes.index(attributes[i])
            patient_data[0, idx] = value
            udm_labels[i].config(text = udm[i])
        except ValueError:
            all_valid = False
            udm_labels[i].config(text = 'Not a number')
            # error_labels[i].config(text = 'Not a number')

    if all_valid:
        if all_binary_values_inserted():
            # print("all OK")
            error_labels[9].config(text=' ')
            result, votes = prediction()
            printOutcome(result, votes)
        else:
            error_labels[9].config(text='You have to select every option')

    # print(patient_data)


def scale_vals(x_ann):
    global sc
    return sc.transform(x_ann)


def prediction():
    global patient_data
    global random_forest
    global decision_tree
    x = patient_data
    x_ann = np.copy(x)
    x_ann = scale_vals(x_ann)
    ann_pred = ann.predict(x_ann)
    ann_pred = (ann_pred > 0.5)
    result, votes = voting(decision_tree.predict(x), random_forest.predict(x), ann_pred)
    return result, votes


def voting(tree_pred, forest_pred, ann_pred):

    result = 0
    votes = np.zeros(shape=(3))
    count = 0

    if ann_pred[0] == 1:
        count += 1
        votes[2] = 1
    if tree_pred[0] == 1:
        count += 1
        votes[0] = 1
    if forest_pred[0] == 1:
        count += 1
        votes[1] = 1
    if count >= 2:
        result = 1

    return result, votes


def config_udm_labels(udm_labels, frame):
    for i in range(len(udm)):
        udm_labels.append(Label(frame))
        udm_labels[i].config(text = udm[i])
        udm_labels[i].grid(row = i, column = 2)


def config_error_labels(error_labels, frame):
    for i in range(len(attributes[:9])):
        error_labels.append(Label(frame))
        error_labels[i].config(text=' ')
        error_labels[i].grid(row=i, column = 2)
    error_labels.append((Label(frame)))
    error_labels[9].config(text = ' ')
    error_labels[9].grid(row=11, column = 3)


def config_labels(attributes, labels, frame):
    for i in range(len(attributes)):
        labels.append(Label(frame))
        labels[i].config(text=attributes[i])
        labels[i].grid(row=i, column=0)


def config_entries(attributes, entries, frame):
    for i in range(len(attributes[:9])):
        entries.append(Entry(frame, width=8))
        entries[i].config()
        entries[i].grid(row=i, column=1)


def config_confirm_buttons(buttons, frame):
    buttons.append(Button(frame, text="Confirm", command=confirmation))
    buttons[10].grid(row=5, column=4)


def config_reset_button(buttons, frame):
    buttons.append(Button(frame, text="Reset", command=reset))
    buttons[11].grid(row=5, column=5)


def setMale():
    global patient_data
    global sex_inserted
    idx = ordered_attributes.index('Sex')
    patient_data[0, idx] = 1
    sex_inserted = True
    global buttons
    buttons[0].config(relief = SUNKEN)
    buttons[1].config(relief=RAISED)


def setFemale():
    global patient_data
    global sex_inserted
    idx = ordered_attributes.index('Sex')
    patient_data[0, idx] = 0
    sex_inserted = True
    global buttons
    buttons[1].config(relief=SUNKEN)
    buttons[0].config(relief=RAISED)


def setDepression():
    global depression_inserted
    global patient_data
    idx = ordered_attributes.index('Depression')
    patient_data[0, idx] = 1
    depression_inserted = True
    global buttons
    buttons[2].config(relief=SUNKEN)
    buttons[3].config(relief=RAISED)


def setNotDepression():
    global patient_data
    global depression_inserted
    idx = ordered_attributes.index('Depression')
    patient_data[0, idx] = 0
    depression_inserted = True
    global buttons
    buttons[3].config(relief=SUNKEN)
    buttons[2].config(relief=RAISED)


def setHTN():
    global HTN_inserted
    global patient_data
    idx = ordered_attributes.index('HTN')
    patient_data[0, idx] = 1
    HTN_inserted = True
    global buttons
    buttons[4].config(relief=SUNKEN)
    buttons[5].config(relief=RAISED)


def setNotHTN():
    global HTN_inserted
    global patient_data
    idx = ordered_attributes.index('HTN')
    patient_data[0, idx] = 0
    HTN_inserted = True
    global buttons
    buttons[5].config(relief=SUNKEN)
    buttons[4].config(relief=RAISED)


def setOA():
    global OA_inserted
    global patient_data
    idx = ordered_attributes.index('OA')
    patient_data[0, idx] = 1
    OA_inserted = True
    global buttons
    buttons[6].config(relief=SUNKEN)
    buttons[7].config(relief=RAISED)

def setNotOA():
    global OA_inserted
    global patient_data
    idx = ordered_attributes.index('OA')
    patient_data[0, idx] = 0
    OA_inserted = True
    global buttons
    buttons[7].config(relief=SUNKEN)
    buttons[6].config(relief=RAISED)

def setCOPD():
    global COPD_inserted
    global patient_data
    idx = ordered_attributes.index('COPD')
    patient_data[0, idx] = 1
    COPD_inserted = True
    global buttons
    buttons[8].config(relief=SUNKEN)
    buttons[9].config(relief=RAISED)


def setNotCOPD():
    global COPD_inserted
    global patient_data
    idx = ordered_attributes.index('COPD')
    patient_data[0, idx] = 0
    COPD_inserted = True
    global buttons
    buttons[9].config(relief=SUNKEN)
    buttons[8].config(relief=RAISED)


def config_binary_buttons(buttons, frame):
    buttons.append(Button(frame, text="Male", command = setMale))
    buttons[0].grid(row=9, column=1)
    buttons.append(Button(frame, text="Female", command = setFemale))
    buttons[1].grid(row=9, column=2)

    buttons.append(Button(frame, text="Yes", command = setDepression))
    buttons[2].grid(row=10, column=1)
    buttons.append(Button(frame, text="No", command = setNotDepression))
    buttons[3].grid(row=10, column=2)

    buttons.append(Button(frame, text="Yes", command = setHTN))
    buttons[4].grid(row=11, column=1)
    buttons.append(Button(frame, text="No", command = setNotHTN))
    buttons[5].grid(row=11, column=2)

    buttons.append(Button(frame, text="Yes", command = setOA))
    buttons[6].grid(row=12, column=1)
    buttons.append(Button(frame, text="No", command = setNotOA))
    buttons[7].grid(row=12, column=2)

    buttons.append(Button(frame, text="Yes", command = setCOPD))
    buttons[8].grid(row=13, column=1)
    buttons.append(Button(frame, text="No", command = setNotCOPD))
    buttons[9].grid(row=13, column=2)


config_labels(attributes, labels, frame)
config_error_labels(error_labels, frame)
config_entries(attributes, entries, frame)
config_udm_labels(udm_labels, frame)
config_binary_buttons(buttons, frame)
config_confirm_buttons(buttons, frame)
config_reset_button(buttons, frame)
reset()

root.mainloop()
