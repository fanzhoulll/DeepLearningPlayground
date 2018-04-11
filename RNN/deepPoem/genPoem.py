from deepPoemUtils import *

def run(usr_input):
    print("Loading text data...")
    text = io.open("./data/shakespeare.txt", encoding='utf-8').read().lower()

    Tx = 40
    chars = sorted(list(set(text)))
    char_indices = dict((c, i) for i, c in enumerate(chars))
    indices_char = dict((i, c) for i, c in enumerate(chars))
    print('number of unique characters in the corpus:', len(chars))

    print("Creating training set...")
    X, Y = build_data(text, Tx, stride = 3)

    print("Vectorizing training set...")
    x, y = vectorization(X, Y, n_x = len(chars), char_indices = char_indices)
    print("Loading model...")
    model = load_model('model/model_shakespeare_kiank.h5')
    # Run this cell to try with different inputs without having to re-train the model
    generate_output(usr_input, model, Tx, chars, char_indices, indices_char)

if __name__== "__main__":
    #usr_input = raw_input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    start = "test"
    run(start)