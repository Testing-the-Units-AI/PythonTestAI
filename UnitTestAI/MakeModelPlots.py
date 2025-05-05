import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
import torch
import re

def plotLossOverEpochs(epochs, train_loss, test_loss, model_name,  model_type=""):
    """
    Creates a plot showing the losses over time for a model.

    :param epochs: The number of epochs the training took place over
    :param train_loss: The losses of training over the epochs
    :param test_loss: The losses of testing over the epochs
    :param name: The name of the trained model being evaluated
    """
    plt.figure(figsize=(10, 6))

    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(model_type + " Loss per Epoch")

    x_range = range(1, epochs + 1)

    plt.plot(x_range, train_loss)
    plt.plot(x_range, test_loss)

    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.plot(x_range, train_loss, label="Training Loss", color='blue')
    plt.plot(x_range, test_loss, label="Testing Loss", color='orange')

    plt.legend()
    plt.savefig(f"./ModelLossCurves/{model_name}.png", dpi = 1200)
    plt.clf()
    plt.close()


TLD ="TrainingSaves"

saves = os.fsencode(f"TrainingSaves")
models = os.listdir(saves)

for model in models:
	model_name = model.decode()
	directory = os.fsencode(f"{TLD}/{model_name}")
	files = os.listdir(directory)

	if len(files) < 1:
		continue

	files = files[1:] + files[0:1]

	test_loss = []
	training_loss = []
	epochs = []

	pattern = re.compile("^Train")
	for file in files:
		filename = os.fsdecode(file)
		if not pattern.match(filename):
			model_save: dict = torch.load(f"{TLD}/{model_name}/{filename}", map_location=torch.device('cpu'))
			test_loss.append(model_save["test_loss"])
			training_loss.append(model_save["train_loss"])
			epochs.append(model_save["epoch"])


	plotLossOverEpochs(
		epochs=epochs+1,
		train_loss=training_loss,
		test_loss=test_loss,
		model_name=model_name
	)

