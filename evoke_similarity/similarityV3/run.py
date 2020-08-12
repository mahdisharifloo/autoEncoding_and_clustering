from compare import Compare
from many_extractor import FeatureExtractor as fe
import pickle as pkl

#create compare object
compOBJ = Compare() 
#import feature from compare objects
#you can find all of vector data that you need 
features = compOBJ.data
# import single image by user : you can use another input methods
single_image_path = input('image path :  ')
product_id = input('input product id :  ')
#create object of feature extractor on many_extractor file to make vector of single image.
fe_obj = fe()
#####################################################
feature_table = fe_obj.feature_table(single_image_path,product_id,'shape','texture','color')
ssim_all = compOBJ.compare(feature_table,'shape','texture','color')

with open('compare_results.pkl','wb') as f:
    pkl.dump(ssim_all,f)



