--[[
Copyright (c) 2016 Michael Wilber

This software is provided 'as-is', without any express or implied
warranty. In no event will the authors be held liable for any damages
arising from the use of this software.

Permission is granted to anyone to use this software for any purpose,
including commercial applications, and to alter it and redistribute it
freely, subject to the following restrictions:

1. The origin of this software must not be misrepresented; you must not
   claim that you wrote the original software. If you use this software
   in a product, an acknowledgement in the product documentation would be
   appreciated but is not required.
2. Altered source versions must be plainly marked as such, and must not be
   misrepresented as being the original software.
3. This notice may not be removed or altered from any source distribution.
--]]

require 'residual-layers'
require 'nn'
require 'data.cifar-dataset'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'train-helpers'

opt = lapp[[
      --batchSize       (default 128)      Sub-batch size
      --dataRoot        (default ./cifar)  Data root folder
      --loadPrefix      (default "")      Model checkpoints path prefix (without rank)
      --maxRank         (default 1)        Max rank to load for averaging
      --startEpoch      (default 1)        Which epoch to start
      --endEpoch        (default 1)        Which epoch to end
]]
print(opt)

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())


cutorch.setDevice(1)
loss = nn.ClassNLLCriterion()
loss:cuda()

function evalModel(epoch, model)
    local train_results = evaluateModel(model, loss, dataTrain, opt.batchSize)
    local test_results = evaluateModel(model, loss, dataTest, opt.batchSize)
    local msg = string.format("epoch = %d, train_loss = %f, train_error = %f, test_loss = %f, test_error = %f", 
    epoch, train_results.loss, 1 - train_results.correct1, test_results.loss, 1.0 - test_results.correct1)
    print(msg)
end

for epoch = opt.startEpoch, opt.endEpoch do
  for r = 1,opt.maxRank do
    local model_file = opt.loadPrefix.."/rank_"..r.."/"..epoch..".model.t7"
    local model = torch.load(model_file)
    evalModel(epoch, model)
    model = nil
    collectgarbage(); collectgarbage(); collectgarbage()
  end
end

