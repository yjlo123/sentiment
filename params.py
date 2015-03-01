#### Trainer parameters

# The database file to train
TRAINER_PARAM_INPUT_FILE_NAME = 'train.csv'

# Only train for the first few records, set to -1 to train the whole database
TRAINER_PARAM_TRAIN_SIZE = 100

# The database file for features
FEATURE_FILE_NAME = 'inquirerbasic.csv'

# Strength threshold for the maximum probability
STRENGTH_THRESHOLD = [0.38, 0.6, 0.88, 0.92, 0.94, 1]

# The database file to tune
DEV_PARAM_INPUT_FILE_NAME = 'test.csv'

# test first few records, set to -1 to test all
DEV_PARAM_TRAIN_SIZE = -1