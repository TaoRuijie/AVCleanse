import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from audiomodel import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.initial_model_a != '':
		print("Model %s loaded from previous state!"%(args.initial_model_a))
		s.load_parameters(args.initial_model_a)
	elif len(args.modelfiles_a) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles_a[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_a[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles_a[-1])

	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.speaker_encoder = ECAPA_TDNN(model = args.model_a).cuda()
		self.speaker_loss    = AAMsoftmax(n_class = args.n_class, m = args.margin_a, s = args.scale_a, c = 192).cuda()	
		self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		print(" Speech model para number = %.2f"%(sum(param.numel() for param in self.speaker_encoder.parameters()) / 1e6))

	def train_network(self, args):
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()

		for num, (speech, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			labels      = torch.LongTensor(labels).cuda()	
			with autocast():
				a_embedding   = self.speaker_encoder.forward(speech.cuda(), aug = True)	
				aloss, _ = self.speaker_loss.forward(a_embedding, labels)		
			scaler.scale(aloss).backward()
			scaler.step(self.optim)
			scaler.update()

			index += len(labels)
			loss += aloss.detach().cpu().numpy()
			time_used = time.time() - time_start
			sys.stderr.write(" [%2d] %.2f%% (est %.1f mins) Lr: %5f, Loss: %.5f\r"%\
			(args.epoch, 100 * (num / args.trainLoader.__len__()), time_used * args.trainLoader.__len__() / num / 60, lr, loss/(num)))
			sys.stderr.flush()
		sys.stdout.write("\n")

		args.score_file.write("%d epoch, LR %f, LOSS %f\n"%(args.epoch, lr, loss/num))
		args.score_file.flush()
		return
		
	def eval_network(self, args):
		self.eval()
		scores_a, labels, res = [], [], []
		embeddings = {}
		lines = open(args.eval_trials).read().splitlines()
		for a_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():
				a_embedding = self.speaker_encoder.forward(a_data[0].cuda())
				for num in range(len(filenames)):
					filename = filenames[num][0]
					a = torch.unsqueeze(a_embedding[num], dim = 0)
					embeddings[filename] = F.normalize(a, p=2, dim=1)
		
		for line in tqdm.tqdm(lines):			
			a1 = embeddings[line.split()[1]]
			a2 = embeddings[line.split()[2]]
			score_a = torch.mean(torch.matmul(a1, a2.T)).detach().cpu().numpy()
			scores_a.append(score_a)
			labels.append(int(line.split()[0]))

		for score in [scores_a]:
			EER = tuneThresholdfromScore(score, labels, [1, 0.1])[1]
			fnrs, fprs, thresholds = ComputeErrorRates(score, labels)
			minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
			res.extend([EER, minDCF])
		
		print('EER_a %2.4f, min_a %.4f\n'%(res[0], res[1]))
		args.score_file.write("EER_a %2.4f, min_a %.4f\n"%(res[0], res[1]))
		args.score_file.flush()
		return

	def save_parameters(self, path):
		model = OrderedDict(list(self.speaker_encoder.state_dict().items()) + list(self.speaker_loss.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			if ('face_encoder.' not in name) and ('face_loss.' not in name):
				if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):			
					if name == 'weight':
						name = 'speaker_loss.' + name
					else:
						name = 'speaker_encoder.' + name
				self_state[name].copy_(param)