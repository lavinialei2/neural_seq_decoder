modelName = 'speechBaseline4'

args = {}
args['outputDir'] = 'logs/' + modelName
args['datasetPath'] = 'ptDecoder_ctc'
args['seqLen'] = 150
args['maxTimeSeriesLen'] = 1200
args['batchSize'] = 32
args['lrStart'] = 0.02
args['lrEnd'] = 0.02
args['nUnits'] = 256
args['nBatch'] = 25000 #3000
args['nLayers'] = 6
args['seed'] = 0
args['nClasses'] = 40
args['nInputFeatures'] = 256
args['dropout'] = 0.4
args['whiteNoiseSD'] = 0.8
args['constantOffsetSD'] = 0.2
args['gaussianSmoothWidth'] = 2.0
args['strideLen'] = 4
args['kernelLen'] = 32
args['bidirectional'] = True
args['l2_decay'] = 1e-5

# added
args['use_layernorm'] = True
args['patience'] = True
args['patience_limit'] = 50
args['model'] = 'LSTM' # GRU, LSTM
args['use_gradClip'] = True
args['gradClip'] = 10
args['use_AdamW'] = True

from neural_decoder.neural_decoder_trainer import trainModel

trainModel(args)
