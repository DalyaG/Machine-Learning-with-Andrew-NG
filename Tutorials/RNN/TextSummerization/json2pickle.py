import _pickle as pickle #data preprocessing
import json
import jsonpickle


def convert_json2pickle(input_file, output_file):

    # open json file
    temp = open(input_file)
    json_data = json.load(temp)
    temp.close()

    # convert to pickle
    JSON_data = jsonpickle.encode(json_data)
    pickled_data = jsonpickle.decode(JSON_data)

    pickle.dump(pickled_data, open(output_file, "wb"))

    with open(output_file, 'rb') as fp:
        head, desc = pickle.load(fp)