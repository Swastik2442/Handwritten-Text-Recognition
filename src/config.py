import string

CHARACTERS = string.ascii_letters + string.digits + string.punctuation + " "
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(CHARACTERS)}
CHAR2IDX['<unk>'] = len(CHAR2IDX) + 1
HEX2CHAR = {hex(ord(char))[2:]: char for char in CHARACTERS}
IDX2CHAR = {i: c for c, i in CHAR2IDX.items()}
NUM_CLASSES = len(CHAR2IDX) + 1

IMAGE_SIZE = (256, 64)
NUM_EPOCHS = 10
