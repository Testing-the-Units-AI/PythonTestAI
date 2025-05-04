from FinalProject import plotLossOverEpochs

training_l = [5, 4, 3, 2, 1]
test_l = [6, 4, 4, 2, 2]

plotLossOverEpochs(5, training_l, test_l, "Cool f***ing model", "Transformer")

training_l = [-10, -5, -2, 0, 2]
test_l = [2, 0, 2, 5, 10]

plotLossOverEpochs(5, training_l, test_l, "Cooler f***ing model", "Transformer")

training_l = [3, 1, 4, 1, 5, 9]
test_l = [9, 1, 5, 4, 1, 3]

plotLossOverEpochs(6, training_l, test_l, "Coolest f***ing model", "Transformer")