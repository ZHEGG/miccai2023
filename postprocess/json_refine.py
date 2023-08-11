import numpy as np
import os
import json

def combine_multimodel(json_dict1,json_dict2,results_dir,team_name = 'workingisallyouneed'):
    json_output_list = []

    with open(json_dict1,'r') as f_r_1:
        json_dict_1 = json.load(f_r_1)
    
    with open(json_dict2,'r') as f_r_2:
        json_dict_2 = json.load(f_r_2)

    for pred_1,pred_2 in zip(json_dict_1,json_dict_2):
        # if pred_1['prediction'] == 1:
        #     pred_info = {
        #     'image_id': pred_1['image_id'],
        #     'prediction': 1,
        #     'score': pred_1['score'],
        #     }
        #     json_output_list.append(pred_info)
        #     continue

        pred_1_score = pred_1['score']
        pred_2_score = pred_2['score']
        # pred_score = [(a+b)/2 for a,b in zip(pred_1_score,pred_2_score)]
        # pred_score = [max(a,b) for a,b in zip(pred_1_score,pred_2_score)]
        if np.array(pred_1_score).argmax() in [3, 4]:
            pred_score =  pred_1_score
        elif np.array(pred_2_score).argmax() in [0, 1]:
            pred_score = pred_2_score
        else:
            pred_score = [(0.5*a+0.5*b) for a,b in zip(pred_1_score,pred_2_score)]

        index = int(np.argmax(np.array(pred_score)))
        pred_info = {
            'image_id': pred_1['image_id'],
            'prediction': index,
            'score': pred_score,
        }
        json_output_list.append(pred_info)

    json_data = json.dumps(json_output_list, indent=4)
    save_name = os.path.join(results_dir, team_name+'_combine.json')
    file = open(save_name, 'w')
    file.write(json_data)
    file.close()

if __name__ == "__main__":
    json_predict_file = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/output/predict_multimodel_submit_v16/trainval.json')
    json_predict_file2 = os.path.join('/home/mdisk3/bianzhewu/medical_repertory/miccai2023/LLD-MMRI2023/main/output/predict_multimodel_submit/ensemble.json')
    root_dir = os.path.dirname(json_predict_file2)

    combine_multimodel(json_predict_file,json_predict_file2,root_dir)