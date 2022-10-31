import glob, numpy, os, random, soundfile, torch, cv2, wave
from scipy import signal
import torchvision.transforms as transforms

def init_loader(args):
	trainloader = train_loader(**vars(args))
	args.trainLoader = torch.utils.data.DataLoader(trainloader, batch_size = args.batch_size, shuffle = True, num_workers = args.n_cpu, drop_last = True)
	evalLoader = eval_loader(**vars(args))
	args.evalLoader = torch.utils.data.DataLoader(evalLoader, batch_size = 1, shuffle = False, num_workers = args.n_cpu, drop_last = False)
	return args

class train_loader(object):
	def __init__(self, train_list, train_path, **kwargs):
		self.train_path = train_path
		self.data_list = []
		self.data_label = []
		lines = open(train_list).read().splitlines()
		dictkeys = list(set([x.split()[0] for x in lines]))		
		dictkeys.sort()
		dictkeys = { key : ii for ii, key in enumerate(dictkeys) }
		for index, line in enumerate(lines):
			speaker_label = dictkeys[line.split()[0]]
			file_name     = line.split()[1]
			self.data_label.append(speaker_label)
			self.data_list.append(file_name)

	def __getitem__(self, index):
		file = self.data_list[index]
		label = self.data_label[index]
		faces    = self.load_face(file = file)
		faces = torch.FloatTensor(numpy.array(faces))
		return faces, label

	def load_face(self, file):
		frames = glob.glob("%s/*.jpg"%(os.path.join(self.train_path, 'frame_align', file[:-4])))
		frame = random.choice(frames)
		frame = cv2.imread(frame)			
		face = cv2.resize(frame, (112, 112))
		face = numpy.array(self.face_aug(face))		
		face = numpy.transpose(face, (2, 0, 1))
		return face

	def __len__(self):
		return len(self.data_list)

	def face_aug(self, face):		
		global_transform = transforms.Compose([
			transforms.ToPILImage(),
			transforms.RandomHorizontalFlip(p=0.5),
			transforms.GaussianBlur(kernel_size=(5, 9),sigma=(0.1, 5)),
			transforms.RandomGrayscale(p=0.2)
		])
		return global_transform(face)

class eval_loader(object):
	def __init__(self, eval_list, eval_path, num_eval_frames = 5, **kwargs):        
		self.data_list, self.data_length = [], []
		self.eval_path = eval_path
		self.num_eval_frames = num_eval_frames
		lines = open(eval_list).read().splitlines()
		for line in lines:
			data = line.split()
			self.data_list.append(data[-2])
			self.data_length.append(float(data[-1]))

		inds = numpy.array(self.data_length).argsort()
		self.data_list, self.data_length = numpy.array(self.data_list)[inds], \
										   numpy.array(self.data_length)[inds]
		self.minibatch = []
		start = 0
		while True:
			frame_length = self.data_length[start]
			minibatch_size = max(1, int(100 // frame_length)) 
			end = min(len(self.data_list), start + minibatch_size)
			self.minibatch.append([self.data_list[start:end], frame_length])
			if end == len(self.data_list):
				break
			start = end

	def __getitem__(self, index):
		data_lists, frame_length = self.minibatch[index]
		filenames, faces = [], []

		for num in range(len(data_lists)):
			file_name = data_lists[num]
			filenames.append(file_name)

			frames = glob.glob("%s/*.jpg"%(os.path.join(self.eval_path, 'frame_align', file_name[:-4])))				
			index = numpy.linspace(0,len(frames) - 1,num=min(self.num_eval_frames, len(frames)))
			if len(index) < self.num_eval_frames:
				index = numpy.pad(index, (0, self.num_eval_frames - len(index)), 'edge')
			frames = [frames[int(i)] for i in index]		
			face = []				
			for frame in frames:
				frame = cv2.imread(frame)				
				frame = cv2.resize(frame, (112, 112))
				frame = numpy.transpose(frame, (2, 0, 1))
				face.append(frame)
			face = numpy.array(face)
			faces.append(face)

		faces = torch.FloatTensor(numpy.array(faces))
		faces = faces.div_(255).sub_(0.5).div_(0.5)
		return faces, filenames

	def __len__(self):
		return len(self.minibatch)