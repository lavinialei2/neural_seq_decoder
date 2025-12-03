modelName = 'speechBaseline_gru_batch128_adamw_lightaug'

args = {}
args['outputDir'] = '/home/lavinialei/neural_seq_decoder/baseline_logs/' + modelName
args['datasetPath'] = '/home/lavinialei/neural_seq_decoder/ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 128
args['lrStart'] = 0.02
args['lrEnd'] = 0.002
args['nUnits'] = 256
args['nBatch'] = 10000 #3000
args['nLayers'] = 5
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.2
args['whiteNoiseSD'] = 0.4
args['constantOffsetSD'] = 0.1
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = False
args['l2_decay'] = 1e-5
args['rnn_type'] = 'gru'
args['label_smoothing'] = 0.1
args['grad_clip'] = 5.0
args['warmup_steps'] = 500
args['time_mask_count'] = 1
args['time_mask_max_length'] = 10
args['eval_every'] = 100
args['early_stopping_patience'] = 50
args['optimizer'] = 'adamw'
args['adam_epsilon'] = 1e-8
args['feature_mask_count'] = 0
args['feature_mask_size'] = 0
args['post_ffn_layers'] = 0
args['post_ffn_hidden'] = 256
args['post_ffn_dropout'] = 0.0

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
