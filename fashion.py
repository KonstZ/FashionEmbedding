# -*- coding: utf-8 -*-
	
import sys
import os
import argparse
import gzip
import json
from collections import Counter
import re

import numpy as np

import cv2

from keras.layers import Dense, Input, Lambda, merge, BatchNormalization, Activation, Dropout, Reshape
from keras.models import Model
from keras.layers import Convolution2D, MaxPooling2D, SpatialDropout2D, Flatten, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
import keras
import keras.backend as K

import tensorflow as tf

from adam import AdamL2

from multiprocessing.pool import ThreadPool

from nltk.stem import WordNetLemmatizer
Wordnet_Lemmatizer = WordNetLemmatizer()

keras.backend.set_image_dim_ordering('tf')

parser = argparse.ArgumentParser(description='Predict image word')
parser.add_argument('--pretrain', action='store_true')
parser.add_argument('--train', action='store_true')
parser.add_argument('--continue_training', action='store_true')
parser.add_argument('--lr', type=float, help = "Learning rate", default = 0.001)
parser.add_argument('--num_epochs', type=int, help = "Epochs to train", default = 10)
parser.add_argument('--convert', action='store_true')
args = parser.parse_args()

DATA_DIR=""#"/hdd2/mode/"
IMAGES_DIR = DATA_DIR + "images/"
MARKUP = DATA_DIR + "markup_agg.tsv.gz"

IMAGE_SIZE=220
BATCH_SIZE=256
WORD_TARGET_SIZE=2048

DROPOUT=0.1

np.random.seed(1337)

def freeze(model):
	frozen = []
	for layer in model.layers:
		if layer.trainable:
			layer.trainable = False
			frozen.append(layer)
		for child in ['layer', 'forward_layer', 'backward_layer']:
			if hasattr(layer, child):
				child = getattr(layer, child)
				if child.trainable:
					child.trainable = False
					frozen.append(child)
	return frozen

def unfreeze(frozen):
	for layer in frozen:
		layer.trainable = True


def get_ids_lables():
	result = set()
	for line in gzip.open(MARKUP):
		if line.startswith('INPUT:'):
			continue
		parts = line.strip().split("\t")
		id = os.path.splitext(parts[2])[0]
		lable = parts[1]
		result.add((id, lable))
		id = os.path.splitext(parts[5])[0]
		lable = parts[4]
		result.add((id, lable))
	return list(result)

Ids_Lables = get_ids_lables()
print "Ids_Lables:", len(Ids_Lables)

STOP_WORDS = set(("the not dont just have has can you are who what such was this new and for get want need from but "+
	"under only with without all see more your yourself our order sell selling sale sales size shop para never better "+
	"sizes shipping ship dhl through time one two good best product birthday price self model keep know drink saying").split())

def parse_lable(lable):
	return set(re.split("[ ,!/]+", lable.strip().lower()))

def is_word(w):
	return len(w) > 2 and w[0].isalpha() and not re.search("\d", w) and not w in STOP_WORDS

def get_words_dict(ids_lables):
	stat = Counter()
	for _, lable in ids_lables:
		words = parse_lable(lable)
		stat.update(words)

	lemm_stat = Counter()
	for w, c in stat.iteritems():
		w = Wordnet_Lemmatizer.lemmatize(w)
		if is_word(w):
			lemm_stat[w] += c

	stat = sorted(lemm_stat.iteritems(), key=lambda x: x[1], reverse=True)[:WORD_TARGET_SIZE]
	print stat[:10]
	print stat[-10:]
	return dict((wc[0], i) for i, wc in enumerate(stat)), \
		np.array([wc[1] for wc in stat], np.float32) / len(ids_lables)	

Word_Dict, Word_Stat = get_words_dict(Ids_Lables)
print "Word_Dict:", len(Word_Dict)

def recode_lable(lable):
	words = parse_lable(lable)
	words = filter(None, map(lambda w : Word_Dict.get(Wordnet_Lemmatizer.lemmatize(w)), words))
	return words

def make_train_test(ids_lables):
	train = []
	test = []
	for image, lable in ids_lables:
		dst = train if hash(image) % 10 != 0 else test
		dst.append((image, recode_lable(lable)))
	return train, test

def sheer(image, angle, axis):
	angle = int(image.shape[axis] * angle)
	if abs(angle) > 1:
		bands = np.rint(np.linspace(0, image.shape[axis], abs(angle))).astype(np.int)
		for i in xrange(1, bands.shape[0]):
			shift = i - angle // 2
			if shift:
				image[bands[i-1]:bands[i]] = np.roll(image[bands[i-1]:bands[i]], shift, axis=axis)

def augment_image(image):
	#resize
	if np.random.random() < 0.1:
		if np.random.random() > 0.5:
			pad1 = np.random.randint(10,20)
			pad2 = np.random.randint(10,20)
			image = np.pad(image, ((pad1,pad1), (pad2,pad2), (0,0)), mode='edge')
		else:
			clip1 = np.random.randint(10,20)
			clip2 = np.random.randint(10,20)
			image = image[clip1:-clip1, clip2:-clip2]
		image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))

	#sheer
	if np.random.random() < 0.2:
		sheer(image, np.random.normal(0, 0.05), axis=0)
	if np.random.random() < 0.2:
		sheer(image, np.random.normal(0, 0.05), axis=1)

	#shift
	if np.random.random() < 0.2:
		shift = np.random.randint(-10,11)
		image = np.roll(image, shift, axis=0)
	if np.random.random() < 0.2:
		shift = np.random.randint(-10,11)
		image = np.roll(image, shift, axis=1)

	#flip
	if np.random.random() < 0.5:
		image = image[:,::-1,:]

	#color
	if np.random.random() < 0.25:
		delta_color = 250*(np.random.normal(0, 0.02) + np.random.normal(0, 0.02, size=(3,)))
		image = np.clip(image + delta_color, 0, 255)
		image = np.rint(image).astype(np.uint8)

	return image

def load_image(image_id, randomize):
	fname = IMAGES_DIR + str(hash(image_id)%100) + "/" + image_id + ".jpeg"
	image = cv2.imread(fname)
	if image is None:
		return None
	if image.shape[0] != IMAGE_SIZE or image.shape[1] != IMAGE_SIZE:
		image = cv2.resize(image, (IMAGE_SIZE,IMAGE_SIZE))
	#bgr -> rgb	
	image = image[...,[2,1,0]]
	if randomize:
		image = augment_image(image)
			
	return image

def data_generator(markup, randomize):
	pool = ThreadPool(4) 
	current = 0
	while True:
		if randomize:
			np.random.shuffle(markup)
		for i in range(len(markup) / BATCH_SIZE):
			for image, lable in pool.imap_unordered(lambda il:(load_image(il[0], randomize), il[1]), markup[i*BATCH_SIZE:(i+1)*BATCH_SIZE]):
				if image is None:
					continue
				if current >= BATCH_SIZE:
					yield image_batch, target_batch
					current = 0
				if current == 0:
					image_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
					target_batch = np.zeros((BATCH_SIZE, WORD_TARGET_SIZE), dtype=np.uint8)
				image_batch[current] = image
				target_batch[current][lable] = 1
				current += 1

def	conv_block(numfilters, size = 1, subsample = 1):
	def f(input):
		conv = Convolution2D(numfilters, size, padding='same', strides = subsample, 
			activation='relu', kernel_initializer='he_normal')(input)
		return conv
	return f

def	pooling_block(subsample = 2):
	return MaxPooling2D(pool_size = subsample, padding='same')

def	darknet_block(numfilters, k, subsample = 1, residual=False):
	def f(input):
		conv3 = conv_block(int(numfilters * k), 3, subsample = subsample)(input)
		conv1 = Dense(numfilters)(conv3)
		if residual:
			return keras.layers.Add()([input, conv1])
		else:
			return LeakyReLU(alpha=0.05)(conv1)
	return f		

def normalize(x):
	return (x - 128.) / 128.

def build_model(dropout=DROPOUT):
	inputLayer = Input(shape=(None, None, 3), dtype='float32')
	x = inputLayer
	x = Lambda(normalize)(x)
	
	x = conv_block(16, 3, subsample = 2)(x)
	x = conv_block(24, 3)(x)
	x = pooling_block(2)(x)
	x = Dropout(dropout)(x)

	x = conv_block(32, 3)(x)
	x = conv_block(32, 3)(x)
	x = pooling_block(2)(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout)(x)

	x = darknet_block(64, 2)(x)
	x = darknet_block(64, 2, residual=True)(x)
	x = pooling_block(2)(x)
	x = BatchNormalization()(x)
	x = Dropout(dropout)(x)

	x = darknet_block(128, 2)(x)
	x = darknet_block(128, 2, residual=True)(x)
	x = Dropout(dropout)(x)
	
	x = conv_block(256, 3)(x)
	x = Dropout(dropout)(x)

	x = GlobalAveragePooling2D()(x)
	result = BatchNormalization()(x)

	return Model(inputs=inputLayer, outputs=result)

def build_word_model(model):
	inputLayer = Input(shape=(None, None, 3), dtype='float32')
	x = inputLayer	
	x = model(x)
	x = Dropout(0.1)(x)

	bias = np.log(Word_Stat)
	result = Dense(WORD_TARGET_SIZE, bias_initializer=keras.initializers.Constant(bias))(x)

	return Model(inputs=inputLayer, outputs=result)

def loss(y_true, y_pred):
	weight = 1 / (1 + tf.log1p(tf.range(tf.shape(y_true)[1])))
	return K.binary_crossentropy(y_true, y_pred, from_logits=True)# * K.expand_dims(weight, axis=0)

def acc(y_true, y_pred):
	return K.cast(K.equal(y_pred > 0, y_true > 0.7), K.floatx()) 

def pr(y_true, y_pred):
	y_true = y_true > 0.7
	y_pred = y_pred > 0
	return K.sum(K.cast(y_pred & y_true, K.floatx())) / (K.epsilon() + K.sum(K.cast(y_pred, K.floatx())))

def rec(y_true, y_pred):
	y_true = y_true > 0.7
	y_pred = y_pred > 0
	return K.sum(K.cast(y_pred & y_true, K.floatx())) / (K.epsilon() + K.sum(K.cast(y_true, K.floatx())))

if args.pretrain:
	train_set, test_set = make_train_test(Ids_Lables)

	backup_dir = "/hdd/models_backup/mode/word"
	if not os.path.exists(backup_dir):
		os.makedirs(backup_dir)
	backup_models = keras.callbacks.ModelCheckpoint(os.path.join(backup_dir,"model.{epoch:03d}"))
	logger = keras.callbacks.CSVLogger("log_pre.csv")

	base_model = build_model()
	word_model = build_word_model(base_model)

	def train(lr, num_epochs):
		word_model.compile(loss=loss, metrics=[pr, rec],
			optimizer=AdamL2(lr=lr, decay=1e-4, clipvalue=2, l2=1e-8))
		word_model.fit_generator(data_generator(train_set, True), steps_per_epoch=len(train_set)//BATCH_SIZE, 
			epochs=num_epochs, 
			verbose=1, 
			callbacks=[backup_models, logger],
			validation_data=data_generator(test_set, False), validation_steps=len(test_set)//BATCH_SIZE)

	if args.continue_training:
		word_model.load_weights("word_model")

	train(lr=args.lr, num_epochs=args.num_epochs)

	base_model.save_weights("base_model")
	word_model.save_weights("word_model")

def make_pair_train_test():
	train = []
	test = []
	for line in gzip.open(MARKUP):
		if line.startswith('INPUT:'):
			continue
		parts = line.strip().split("\t")

		image1 = os.path.splitext(parts[2])[0]
		image2 = os.path.splitext(parts[5])[0]

		target = 1 if parts[6] == 'yes' else 0
		jtarget = json.loads(parts[7]) 
		if jtarget["likes"] == 3 and jtarget["votes"] == 5:
			continue
		#target = float(jtarget["likes"]) / jtarget["votes"] 
		
		if hash(image1) % 10 != 0 and hash(image2) % 10 != 0:
			train.append((image1, image2, target))
		elif hash(image1) % 10 == 0 and hash(image2) % 10 == 0:
			test.append((image1, image2, target))
		
	print len(train), len(test)
	return train, test

def load_pair(markup, randomize):
	image1, image2, match = markup
	image1 = load_image(image1, randomize)
	image2 = load_image(image2, randomize)
	return image1, image2, match

def match_generator(markup, randomize):
	pool = ThreadPool(4) 
	current = 0
	while True:
		if randomize:
			np.random.shuffle(markup)
		for i in range(len(markup) / BATCH_SIZE):
			for image1, image2, match in pool.imap_unordered(lambda m:load_pair(m, randomize), markup[i*BATCH_SIZE:(i+1)*BATCH_SIZE]):
				if image1 is None or image2 is None:
					continue
				if current >= BATCH_SIZE:
					yield [image1_batch, image2_batch], target_batch
					current = 0
				if current == 0:
					image1_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
					image2_batch = np.zeros((BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.float32)
					target_batch = np.zeros((BATCH_SIZE, 1), dtype=np.float32)
				image1_batch[current] = image1
				image2_batch[current] = image2
				target_batch[current] = match
				current += 1

def distance(x):
	return 1 - K.sum(K.prod(x, axis=1), axis=-1, keepdims=True)

def build_embedding_model(model):
	inputLayer = Input(shape=(None, None, 3), dtype='float32')
	x = model(inputLayer)
	x = Dense(128, activation='tanh')(x)
	result = Lambda(lambda x : K.l2_normalize(x, axis=-1))(x)

	return Model(inputs=inputLayer, outputs=result)

def build_match_model(model):
	input1 = Input(shape=(None, None, 3), dtype='float32')
	input2 = Input(shape=(None, None, 3), dtype='float32')

	x1 = model(input1)
	x2 = model(input2)

	x = Lambda(lambda xx : tf.stack(xx, axis=1))([x1,x2])
	x = Dropout(0.1, noise_shape=(BATCH_SIZE, 1, 128))(x)
	result = Lambda(distance)(x)

	return Model(inputs=[input1, input2], outputs=result)

def mloss(y_true, y_pred):
	y_true = K.cast(y_true > 0.7, K.floatx()) 
	return K.relu(0.5 + (2 * y_true - 1) * (y_pred - 0.5))

def mpr(y_true, y_pred):
	y_true = y_true > 0.7
	y_pred = y_pred < 0.5
	return K.sum(K.cast(y_pred & y_true, K.floatx())) / (K.epsilon() + K.sum(K.cast(y_pred, K.floatx())))

def mrec(y_true, y_pred):
	y_true = y_true > 0.7
	y_pred = y_pred < 0.5
	return K.sum(K.cast(y_pred & y_true, K.floatx())) / (K.epsilon() + K.sum(K.cast(y_true, K.floatx())))

if args.train:
	train_set, test_set = make_pair_train_test()

	backup_dir = "/hdd/models_backup/mode/"
	if not os.path.exists(backup_dir):
		os.makedirs(backup_dir)
	backup_models = keras.callbacks.ModelCheckpoint(os.path.join(backup_dir,"model.{epoch:03d}"))
	logger = keras.callbacks.CSVLogger("log.csv")

	base_model = build_model()
	embedding_model = build_embedding_model(base_model)
	match_model = build_match_model(embedding_model)

	def train(lr, num_epochs):
		match_model.compile(loss=mloss, metrics=[mpr, mrec],
			optimizer=AdamL2(lr=lr, decay=1e-4, clipvalue=2, l2=1e-8))
		match_model.fit_generator(match_generator(train_set, True), steps_per_epoch=len(train_set)//BATCH_SIZE, 
			epochs=num_epochs, 
			verbose=1, 
			callbacks=[backup_models, logger],
			validation_data=match_generator(test_set, False), validation_steps=len(test_set)//BATCH_SIZE)

	if args.continue_training:
		match_model.load_weights("match_model")
	else:
		base_model.load_weights("base_model")
		frozen = freeze(base_model)
		train(lr=args.lr, num_epochs=1)
		unfreeze(frozen)

#	train(lr=args.lr/10, num_epochs=args.num_epochs)

	base_model.save_weights("match_image_model")
	embedding_model.save_weights("embedding_model")
	match_model.save_weights("match_model")

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io

def convert():
	K.set_learning_phase(0)

	base_model = build_model()
	model = build_embedding_model(base_model)
	model.load_weights("embedding_model")
	model._make_predict_function()
	print model.inputs
	print model.outputs
 
	sess = K.get_session()
 
	constant_graph = sess.graph.as_graph_def()
	constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), 
		['lambda_2/l2_normalize'])
	graph_io.write_graph(constant_graph, '', 'model.pb', as_text=False)        

if args.convert:
	DROPOUT=0
	convert()

