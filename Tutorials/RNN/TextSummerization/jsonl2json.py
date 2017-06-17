import json
import jsonlines


def convert_jsonl2json(input_file, output_file):

    # open jsonl file
    temp = open(input_file)
    jsonl_data = jsonlines.Reader(temp)

    # prepare file for json
    json_data = []

    # extract wanted fields
    for item in jsonl_data:
        json_data.append({'desc':item['content'], 'head':item['title']})

    # close readers
    jsonl_data.close()
    temp.close()

    # write to json file
    with open(output_file, 'w') as outfile:
        json.dump(json_data, outfile, indent=4)