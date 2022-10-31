import torch, sys, os, tqdm, numpy, soundfile, time, pickle, cv2, glob, random, scipy
import torch.nn as nn
from tools import *
from loss import *
from visualmodel import *
from collections import defaultdict, OrderedDict
from torch.cuda.amp import autocast,GradScaler

def init_trainer(args):
	s = trainer(args)
	args.epoch = 1
	if args.initial_model_v != '':
		print("Model %s loaded from previous state!"%(args.initial_model_v))
		s.load_parameters(args.initial_model_v)
	elif len(args.modelfiles_v) >= 1:
		print("Model %s loaded from previous state!"%args.modelfiles_v[-1])
		args.epoch = int(os.path.splitext(os.path.basename(args.modelfiles_v[-1]))[0][6:]) + 1
		s.load_parameters(args.modelfiles_v[-1])
	return s

class trainer(nn.Module):
	def __init__(self, args):
		super(trainer, self).__init__()
		self.face_encoder    = IResNet(model = args.model_v).cuda()
		self.face_loss       = AAMsoftmax(n_class =  args.n_class, m = args.margin_v, s = args.scale_v, c = 512).cuda()
		self.optim           = torch.optim.Adam(self.parameters(), lr = args.lr, weight_decay = 2e-5)
		self.scheduler       = torch.optim.lr_scheduler.StepLR(self.optim, step_size = args.test_step, gamma = args.lr_decay)
		print(" Face model para number = %.2f"%(sum(param.numel() for param in self.face_encoder.parameters()) / 1e6))

	def train_network(self, args):
		self.train()
		scaler = GradScaler()
		self.scheduler.step(args.epoch - 1)
		index, top1, loss = 0, 0, 0
		lr = self.optim.param_groups[0]['lr']
		time_start = time.time()

		for num, (face, labels) in enumerate(args.trainLoader, start = 1):
			self.zero_grad()
			labels      = torch.LongTensor(labels).cuda()	
			face        = face.div_(255).sub_(0.5).div_(0.5)
			with autocast():
				v_embedding   = self.face_encoder.forward(face.cuda())	
				vloss, _ = self.face_loss.forward(v_embedding, labels)			
			scaler.scale(vloss).backward()
			scaler.step(self.optim)
			scaler.update()

			index += len(labels)
			loss += (vloss).detach().cpu().numpy()
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
		scores_v, labels, res = [], [], []
		embeddings = {}
		lines = open(args.eval_trials).read().splitlines()
		for v_data, filenames in tqdm.tqdm(args.evalLoader, total = len(args.evalLoader)):
			with torch.no_grad():
				v_data = v_data[0].transpose(0, 1)
				v_outs = []
				for i in range(v_data.shape[0]):
					v_outs.append(self.face_encoder.forward(v_data[i].cuda()))
				v_embedding = torch.stack(v_outs)
				for num in range(len(filenames)):
					filename = filenames[num][0]
					v = v_embedding[:,num,:]
					embeddings[filename] = F.normalize(v, p=2, dim=1)
		
		for line in tqdm.tqdm(lines):			
			v1 = embeddings[line.split()[1]]
			v2 = embeddings[line.split()[2]]
			score_v = torch.mean(torch.matmul(v1, v2.T)).detach().cpu().numpy()				
			scores_v.append(score_v)
			labels.append(int(line.split()[0]))

		for score in [scores_v]:
			EER = tuneThresholdfromScore(score, labels, [1, 0.1])[1]
			fnrs, fprs, thresholds = ComputeErrorRates(score, labels)
			minDCF, _ = ComputeMinDcf(fnrs, fprs, thresholds, 0.05, 1, 1)
			res.extend([EER, minDCF])
		
		print('EER_v %2.4f, min_v %.4f\n'%(res[0], res[1]))
		args.score_file.write("EER_v %2.4f, min_v %.4f\n"%(res[0], res[1]))
		args.score_file.flush()
		return

	def save_parameters(self, path):
		model = OrderedDict(list(self.face_encoder.state_dict().items()) + list(self.face_loss.state_dict().items()))
		torch.save(model, path)

	def load_parameters(self, path):
		self_state = self.state_dict()
		loaded_state = torch.load(path)
		for name, param in loaded_state.items():
			if ('speaker_encoder.' not in name) and ('speaker_loss.' not in name):					
				if ('face_encoder.' not in name) and ('face_loss.' not in name):			
					if name == 'weight':
						name = 'face_loss.' + name
					else:
						name = 'face_encoder.' + name
				self_state[name].copy_(param)
			