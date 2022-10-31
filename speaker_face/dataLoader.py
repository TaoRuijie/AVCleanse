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
	def __init__(self, train_list, train_path, musan_path, rir_path, frame_len, **kwargs):
		self.train_path = train_path
		self.frame_len = frame_len * 160 + 240
		self.noisetypes = ['noise','speech','music']
		self.noisesnr = {'noise':[0,15],'speech':[13,20],'music':[5,15]}
		self.numnoise = {'noise':[1,1], 'speech':[3,8], 'music':[1,1]}
		self.noiselist = {}
		augment_files   = glob.glob(os.path.join(musan_path,'*/*/*/*.wav'))
		for file in augment_files:
			if file.split('/')[-4] not in self.noiselist:
				self.noiselist[file.split('/')[-4]] = []
			self.noiselist[file.split('/')[-4]].append(file)
		self.rir_files  = glob.glob(os.path.join(rir_path,'*/*/*.wav'))
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
		segments = self.load_wav(file = file)
		segments = torch.FloatTensor(numpy.array(segments))
		faces    = self.load_face(file = file)
		faces = torch.FloatTensor(numpy.array(faces))
		return segments, faces, label

	def load_wav(self, file):
		utterance, _ = soundfile.read(os.path.join(self.train_path, 'wav', file))
		if utterance.shape[0] <= self.frame_len:
			shortage = self.frame_len - utterance.shape[0]
			utterance = numpy.pad(utterance, (0, shortage), 'wrap')
		startframe = random.choice(range(0, utterance.shape[0] - (self.frame_len)))
		segment = numpy.expand_dims(numpy.array(utterance[int(startframe):int(startframe)+self.frame_len]), axis = 0)
		augtype = random.randint(0,4)
		if augtype == 0:   # Original
			segment = segment
		elif augtype == 1:
			segment = self.add_rev(segment, length = self.frame_len)
		elif augtype == 2:
			segment = self.add_noise(segment, 'speech', length = self.frame_len)
		elif augtype == 3: 
			segment = self.add_noise(segment, 'music', length = self.frame_len)
		elif augtype == 4:
			segment = self.add_noise(segment, 'noise', length = self.frame_len)
		return segment[0]

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

	def add_rev(self, audio, length):
		rir_file    = random.choice(self.rir_files)
		rir, sr     = soundfile.read(rir_file)
		rir         = numpy.expand_dims(rir.astype(numpy.float),0)
		rir         = rir / numpy.sqrt(numpy.sum(rir**2))
		return signal.convolve(audio, rir, mode='full')[:,:length]

	def add_noise(self, audio, noisecat, length):
		clean_db    = 10 * numpy.log10(numpy.mean(audio ** 2)+1e-4)
		numnoise    = self.numnoise[noisecat]
		noiselist   = random.sample(self.noiselist[noisecat], random.randint(numnoise[0],numnoise[1]))
		noises = []
		for noise in noiselist:
			noiselength = wave.open(noise, 'rb').getnframes()
			if noiselength <= length:
				noiseaudio, _ = soundfile.read(noise)
				noiseaudio = numpy.pad(noiseaudio, (0, length - noiselength), 'wrap')
			else:
				start_frame = numpy.int64(random.random()*(noiselength-length))
				noiseaudio, _ = soundfile.read(noise, start = start_frame, stop = start_frame + length)
			noiseaudio = numpy.stack([noiseaudio],axis=0)
			noise_db = 10 * numpy.log10(numpy.mean(noiseaudio ** 2)+1e-4) 
			noisesnr   = random.uniform(self.noisesnr[noisecat][0],self.noisesnr[noisecat][1])
			noises.append(numpy.sqrt(10 ** ((clean_db - noise_db - noisesnr) / 10)) * noiseaudio)
		noise = numpy.sum(numpy.concatenate(noises,axis=0),axis=0,keepdims=True)
		return noise + audio

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
		filenames, segments, faces = [], [], []

		for num in range(len(data_lists)):
			file_name = data_lists[num]
			filenames.append(file_name)

			audio, sr = soundfile.read(os.path.join(self.eval_path, 'wav', file_name))
			if len(audio) < int(frame_length * sr):
				shortage    = int(frame_length * sr) - len(audio) + 1
				audio       = numpy.pad(audio, (0, shortage), 'wrap')
			audio = numpy.array(audio[:int(frame_length * sr)])
			segments.append(audio)

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

		segments = torch.FloatTensor(numpy.array(segments))
		faces = torch.FloatTensor(numpy.array(faces))
		faces = faces.div_(255).sub_(0.5).div_(0.5)
		return segments, faces, filenames

	def __len__(self):
		return len(self.minibatch)