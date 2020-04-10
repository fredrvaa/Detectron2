import json, os

def fix_annotations(annotation_dir):
    for json_file in os.listdir(annotation_dir):
        with open(os.path.join(annotation_dir, json_file)) as file:
            coco_data = json.load(file)
        
        for i, annotation in enumerate(coco_data['annotations']):
            annotation['id'] = i
            annotation['iscrowd'] = 0
            annotation['category_id'] += 1
            bbox = annotation['bbox']
            annotation['area'] = bbox[2] * bbox[3]

        for category in coco_data['categories']:
            category['id'] += 1
        
        with open(os.path.join(annotation_dir, json_file), 'w') as file:
            json.dump(coco_data, file)

