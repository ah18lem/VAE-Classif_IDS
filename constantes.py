
NUM_CLIENTS=10
FRACTION_FIT=1  
FRACTION_EVALUATE=1  
MIN_FIT_CLIENTS=10 
MIN_EVALUATE_CLIENTS=10
MIN_AVAILABLE_CLIENTS=10
VAE=True
BATCH_SIZE=64
LEARNING_RATE=0.0001
 #----------BOT_IoT--------------# 

#TRAINING_DATA="train_bot-iot.csv"
#TESTING_DATA="test_bot-iot.csv"
#NUM_CLASSES=5
#RATIO_LABEL=0.1
#EPOCHS_CLIENT=5
#EPOCHS_SERVEUR=20
#MULTICLASS_TARGET_COL="category"
#ONE_HOT_ENCODING_LIST=['proto']
#DELETE_LIST=["subcategory","saddr" ,"daddr"]
#ENCODER_LAYERS=[14,14,10]
#DECODER_LAYERS=[14,14]
#NUM_ROUNDS=30


#----------Wustl-2020--------------# 
# TRAINING_DATA="train_wustl.csv"
# TESTING_DATA="test_wustl.csv"
# NUM_CLASSES=3
# RATIO_LABEL=0.3
# EPOCHS_CLIENT=2
# EPOCHS_SERVEUR=1
# ENCODER_LAYERS=[30,30,20]
# DECODER_LAYERS=[30,30]
# NUM_ROUNDS=3
# MULTICLASS_TARGET_COL="Attack Category"

#----------SCADA--------------# 

TRAINING_DATA="train_scada.csv"
TESTING_DATA="test_scada.csv"
NUM_CLASSES=8
RATIO_LABEL=0.3
EPOCHS_CLIENT=20
EPOCHS_SERVEUR=100
ENCODER_LAYERS=[20,20,15]
DECODER_LAYERS=[20,20]
NUM_ROUNDS=17
MULTICLASS_TARGET_COL="result"







