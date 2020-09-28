import torch
from torchvision import transforms

class ToTensor():

	def __call__(self, sample):
		sample['x'] = torch.tensor(sample['x'])

		return sample


class Reshape():

	def __init__(self, h, w=None):
		self.h, self.w = h, w
		if not w:
			self.w = h

	def __call__(self, sample):
		sample['x'] = sample['x'].view(-1, self.h, self.w)

		return sample


trans_func = transforms.Compose([
	ToTensor(),
	Reshape(28)
])
