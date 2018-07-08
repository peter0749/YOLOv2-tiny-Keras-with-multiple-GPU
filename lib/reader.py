import config
from pycocotools.coco import COCO
'''
An utility to read all filepaths of images and their masks
'''

def dataset_filepath(root, anno_path, get_masks=True):
    '''
    input: a full path to root of images, annotation path
    output: a list of filepath of images and their masks
    '''
    coco = COCO(anno_path)
    imgIds = coco.getImgIds()
    data = []
    for imgId in imgIds:
        img = coco.loadImgs(imgId)[0]
        annIds = coco.getAnnIds(imgIds=imgId)
        image = {'image': str( root+'/'+img['file_name'] )}
        image['width'], image['height'] = img['width'], img['height']

        if get_masks:
            image['masks'] = []
            annos = coco.loadAnns(annIds)
            for anno in annos:
                x, y, w, h = anno['bbox']
                class_id = anno['category_id']
                image['masks'].append({'id': anno['id'], 'ymin': y, 'xmin': x, 'ymax': y+h, 'xmax': x+w, 'class': config.ID_MAP[class_id]})
        data.append(image)
    return data, coco

if __name__ == '__main__':
    data, coco = dataset_filepath('../../coco/images/val2014', '../../coco/annotations/instances_val2014.json')
    print(data)
