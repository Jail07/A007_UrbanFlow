import torch

# Загружаем содержимое
data = torch.load('hybrid_brain.pth', map_location=torch.device('cpu'))

# Смотрим, что внутри (обычно это словарь/OrderedDict)
print(data)