
import argparse
import pyhocon

from src.dataCenter import *
from src.utils import *
from src.models import *
from src.pixelGan import pixGan
from src.models_enhance import GraphSageEnhanced

parser = argparse.ArgumentParser(description='pytorch version of GraphSAGE')

parser.add_argument('--dataSet', type=str, default='cora')
parser.add_argument('--agg_func', type=str, default='MEAN')
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--b_sz', type=int, default=11)
parser.add_argument('--seed', type=int, default=824)
parser.add_argument('--cuda', action='store_true',default=True,
					help='use CUDA')
parser.add_argument('--gcn', action='store_true')
parser.add_argument('--learn_method', type=str, default='sup')
parser.add_argument('--unsup_loss', type=str, default='normal')
parser.add_argument('--max_vali_f1', type=float, default=0)
parser.add_argument('--name', type=str, default='debug')
parser.add_argument('--config', type=str, default='../experiments.conf') # ../src/
args = parser.parse_args()

'''
pix2pixcel  Emb 生成
应该 通过 SMOTE 或者 AdaSyn 啥的 去生成
不同数据集测试
'''
if torch.cuda.is_available():
	if not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

device = torch.device("cuda" if args.cuda else "cpu")
print('DEVICE:', device)
targetLabel = 1  # cora 1 pubmed label = 0 blog

val_macro = .0
val_micro = .0
test_macro = .0
test_micro = .0
val_auc = .0
val_roc_auc = .0
test_auc = .0
test_roc_auc = .0


if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	# load config file
	config = pyhocon.ConfigFactory.parse_file(args.config)

	# load data
	ds = args.dataSet
	dataCenter = DataCenter(config)
	tttt = [1.]
	PORTION = 1.
	imb_ratio = .5
	for tt in tttt:
		PORTION = tt
		print(PORTION)

		#dataCenter.load_dataSet(ds)
		dataCenter.load_dataSet(ds, targetLabel, imb_ratio=imb_ratio)
		labels_roc_auc = []
		for i in set(dataCenter.labels):
			labels_roc_auc.append(i)

		features = torch.FloatTensor(getattr(dataCenter, ds+'_feats')).to(device)

		graphSage = GraphSage(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features, getattr(dataCenter, ds+'_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
		graphSage.to(device)

		num_labels = len(set(getattr(dataCenter, ds+'_labels')))
		classification = Classification(config['setting.hidden_emb_size'], num_labels)
		classification.to(device)

		unsupervised_loss = UnsupervisedLoss(getattr(dataCenter, ds+'_adj_lists'), getattr(dataCenter, ds+'_train'), device)


		graphSageEnhanced = GraphSageEnhanced(config['setting.num_layers'], features.size(1), config['setting.hidden_emb_size'], features,
							  getattr(dataCenter, ds + '_adj_lists'), device, gcn=args.gcn, agg_func=args.agg_func)
		graphSageEnhanced.to(device)

		classificationEnhanced = Classification(config['setting.hidden_emb_size'], num_labels)
		classificationEnhanced.to(device)
		maxEpoch = args.epochs
		#f = open('targetEmb.txt', 'w')

		if args.learn_method == 'sup':
			print('GraphSage with Supervised Learning')
		elif args.learn_method == 'plus_unsup':
			print('GraphSage with Supervised Learning plus Net Unsupervised Learning')
		else:
			print('GraphSage with Net Unsupervised Learning')

		for epoch in range(args.epochs):
			print('----------------------EPOCH %d-----------------------' % epoch)
			graphSage, classification = apply_model(dataCenter, ds, graphSage, classification, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method, epoch, maxEpoch)

			if 1: # 这个先不用评估，看情况
				if (epoch+1) % 2 == 0 and args.learn_method == 'unsup':
					classification, args.max_vali_f1 = train_classification(dataCenter, graphSage, classification, ds, device, args.max_vali_f1, args.name)
				if args.learn_method != 'unsup':
					val_macro, val_micro,  val_auc, val_roc_auc, test_macro, test_micro, test_auc,test_roc_auc = evaluate(dataCenter, ds, graphSage, classification, device, val_macro, val_micro, val_auc, val_roc_auc, test_macro, test_micro, test_auc,test_roc_auc,args.name, epoch, labels_roc_auc)
		# TEST
		z = 1
		print("Original Test AUC F1:", test_auc)
		print("Original Test AUC_ROC F1:", test_roc_auc)
		print("Original Test macro F1:", test_macro)
		print("Original Test micro F1:", test_micro)

		#torch.save(graphSage, 'models/graphSage_wiki.torch')
		#exit()
		#graphSage = torch.load('models/graphSage_pubmed.torch')


		val_macro = .0
		val_micro = .0
		test_macro = .0
		test_micro = .0
		val_auc = .0
		val_roc_auc = .0
		test_auc = .0
		test_roc_auc = .0

		#minEmb, minIdex, minNeighs, maxLen, dataDim = checkUnSunEmb(dataCenter, ds, graphSage, args.b_sz, targetLabel)

		minEmb, minIdex, minNeighs, maxLen, dataDim, minRecord, recordEmb = checkUnSunEmbAdaSyn2(dataCenter, ds, graphSage, args.b_sz, targetLabel, portion=PORTION)
		# 进行解码
		print ('maxLen ' + str((maxLen)))
		p = pixGan(minEmb, minIdex, minNeighs, maxLen, dataDim)
		p.train()
		p.save()
		z = 1
		minEnhanced = p.generateAdaSynCora(recordEmb)

		z = 1
		enhancedLength = minEnhanced[0].size()[0] # ADAGAN 里面的
		#enhancedLabel = torch.full((enhancedLength, 1), targetLabel)
		enhancedLabel = []
		for j in range(0, enhancedLength):
			enhancedLabel.append(targetLabel)
		enhancedLabel = np.array(enhancedLabel)

		minNodeSelfFeature2 = torch.index_select(features, dim=0, index=torch.LongTensor(minRecord).to(device))


		for epoch in range(args.epochs ):

			graphSageEnhanced, classificationEnhanced = apply_model_enhanced(dataCenter, ds, graphSageEnhanced, classificationEnhanced, unsupervised_loss, args.b_sz, args.unsup_loss, device, args.learn_method,
															 minEnhanced, minNodeSelfFeature2, enhancedLabel, epoch, maxEpoch)

			# Supervised 直接走 evaluate
			val_macro, val_micro, val_auc, val_roc_auc, test_macro, test_micro, test_auc, test_roc_auc = evaluateEnhanced(dataCenter, ds, graphSageEnhanced, classificationEnhanced, device, val_macro, val_micro, val_auc, val_roc_auc, test_macro, test_micro, test_auc,test_roc_auc,args.name, epoch, labels_roc_auc)

		print("Enhanced Test AUC F1:", test_auc)
		print("Enhanced Test AUC_ROC F1:", test_roc_auc)
		print("Enhanced Test macro F1:", test_macro)
		print("Enhanced Test micro F1:", test_micro)

