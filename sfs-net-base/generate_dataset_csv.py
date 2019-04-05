from data_loading import generate_sfsnet_data_csv, generate_celeba_data_csv, generate_celeba_synthesize_data_csv
'''
# SfSNet dataset
dataset_path = '/nfs/bigdisk/bsonawane/sfsnet_data/'
# dataset_path = '../data/sfs-net/'
generate_sfsnet_data_csv(dataset_path + 'train/', dataset_path + '/train.csv')
generate_sfsnet_data_csv(dataset_path + 'test/', dataset_path + '/test.csv')

# CelebA dataset
dataset_path = '/nfs/bigdisk/bsonawane/CelebA-dataset/CelebA_crop_resize_128/'
# dataset_path = '../data/celeba/'
generate_celeba_data_csv(dataset_path + 'train/', dataset_path + '/train.csv')
generate_celeba_data_csv(dataset_path + 'test/', dataset_path + '/test.csv')
'''
dataset_path = '/nfs/bigdisk/bsonawane/CelebA-dataset/CelebA_crop_resize_128/sfs_synthesized_data/'
# dataset_path = '../data/celeba/synthesized_data/'
generate_celeba_synthesize_data_csv(dataset_path + 'train/', dataset_path + '/train.csv')
generate_celeba_synthesize_data_csv(dataset_path + 'test/', dataset_path + '/test.csv')

