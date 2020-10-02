import json

def get_meta_data(fmu_model, output_name):

        name_meanings = {}
        key_values = fmu_model.get_model_variables().keys()

        for key in key_values:
                name_meanings[key] = fmu_model.get_variable_description(key)

        with open(output_name, 'w') as fp:
                json.dump(name_meanings, fp, indent=4)