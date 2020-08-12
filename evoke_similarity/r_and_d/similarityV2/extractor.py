
from many_extractorV2 import FeatureExtractor as ext


ext_obj = ext('dataset/data')

features , labels , image_path_list = ext_obj.extractor()

ext_obj.save(features,image_path_list,feature_file_name='test.pkl')