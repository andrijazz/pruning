from easydict import EasyDict as edict
from torchvision import transforms

C = edict()

# Alias
cfg = C

# Model name
C.MODEL = "BasicModel"

# Learner to use
C.LEARNER = "Learner"

# GPU to use
C.GPU = 0

# conf
C.ARCH = [1000, 1000, 500, 200, 10]

# Train options
C.TRAIN = edict()

# Dataset for training
C.TRAIN.DATASET = edict()
C.TRAIN.DATASET.NAME = "MNIST"
C.TRAIN.DATASET.TRANSFORM = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  (0.1307,), (0.3081,))
    ])

# Step period to write summaries to tensorboard
C.TRAIN.SUMMARY_FREQ = 10
# Step period to perform validation
C.TRAIN.VAL_FREQ = 200
# Train batch size
C.TRAIN.BATCH_SIZE = 20
# Number of epochs
C.TRAIN.NUM_EPOCHS = 2
# Restore from wandb or local
C.TRAIN.RESTORE_STORAGE = 'local'
# Path to checkpoint from which to be restored
C.TRAIN.RESTORE_FILE = ''
# Learning rate
C.TRAIN.LR = 1e-4
# save checkpoint model frequency
C.TRAIN.SAVE_MODEL_FREQ = 5000

# # Validation options.
C.VAL = edict()
# Validation batch size
C.VAL.BATCH_SIZE = 20

# Test options
C.TEST = edict()
# Test batch size
C.TEST.BATCH_SIZE = 20
# Path to checkpoint from which to be restored
C.TEST.RESTORE_FILE = "andrijazz/pruning/2w6vyy0x/basicmodel-5600.pth"
# Restore from wandb or local
C.TEST.RESTORE_STORAGE = 'wandb'
# percentage of weights/units to be pruned
C.TEST.PRUNING_K = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]
# Dataset for testing
C.TEST.DATASET = edict()
C.TEST.DATASET.NAME = "MNIST"
C.TEST.DATASET.TRANSFORM = transforms.Compose([
        # transforms.Resize((224, 224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor()
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  (0.1307,), (0.3081,))
    ])
# Path to h5 file to save results
C.TEST.OUTPUT_FILE = ""
