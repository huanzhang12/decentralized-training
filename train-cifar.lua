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
local nninit = require 'nninit'
local posix = require 'posix'
local pretty_write = require 'pl.pretty'.write

-- Feel free to comment these out.
--[[
hasWorkbook, labWorkbook = pcall(require, 'lab-workbook')
if hasWorkbook then
  workbook = labWorkbook:newExperiment{}
  lossLog = workbook:newTimeSeriesLog("Training loss",
                                      {"nImages", "loss"},
                                      100)
  errorLog = workbook:newTimeSeriesLog("Testing Error",
                                       {"nImages", "error"})
else
  print "WARNING: No workbook support. No results will be saved."
end
--]]

opt = lapp[[
      --batchSize       (default 128)      Sub-batch size
      --maxIter         (default 200)      How many iterations to run
      --iterSize        (default 1)       How many sub-batches in each batch
      --Nsize           (default 3)       Model has 6*n+2 layers.
      --dataRoot        (default ./cifar) Data root folder
      --loadFrom        (default "")      Model to load
      --experimentName  (default "snapshots/cifar-residual-experiment1")
      --dstalg          (default "dstsgd")    Distributed algorithm ("dstsgd", "easgd")
      --nodesFile       (default 'nodes.txt')    A text file with all host names and port number
      --weightsFile     (default 'weights.txt')  A text file with weights for parameters from different machines
      --nodeID          (default 0)              Which node is this machine? Set 0 for auto
      --chunkSize       (default 8192)           TCP-IP transfer chunk size (important to maximize transfer rate)
      --dynBatchSize    (default 0)              Set to 1 to use dynamic batch size
      --saveDir         (default ".")            Model checkpoints path
      --noEval                                   Do not evaluate, only fakeloss will be printed
      --lrDecay1        (default 80)             first learning rate decay > epoch
      --lrDecay2        (default 120)            second learning rate decay > epoch
]]
print(opt)

if opt.dstalg == "dstsgd" then
  require 'train-dstsgd'
  -- we will change this dynamically
  opt.iterSize = 9999999
end
-- rank of this machine
self_rank = 1

-- create data loader
dataTrain = Dataset.CIFAR(opt.dataRoot, "train", opt.batchSize)
dataTest = Dataset.CIFAR(opt.dataRoot, "test", opt.batchSize)
local mean,std = dataTrain:preprocess()
dataTest:preprocess(mean,std)
print("Dataset size: ", dataTrain:size())


-- Residual network.
-- Input: 3x32x32
local N = opt.Nsize
if opt.loadFrom == "" then
    input = nn.Identity()()
    ------> 3, 32,32
    model = cudnn.SpatialConvolution(3, 16, 3,3, 1,1, 1,1)
                :init('weight', nninit.kaiming, {gain = 'relu'})
                :init('bias', nninit.constant, 0)(input)
    model = cudnn.SpatialBatchNormalization(16)(model)
    model = cudnn.ReLU(true)(model)
    ------> 16, 32,32   First Group
    for i=1,N do   model = addResidualLayer2(model, 16)   end
    ------> 32, 16,16   Second Group
    model = addResidualLayer2(model, 16, 32, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 32)   end
    ------> 64, 8,8     Third Group
    model = addResidualLayer2(model, 32, 64, 2)
    for i=1,N-1 do   model = addResidualLayer2(model, 64)   end
    ------> 10, 8,8     Pooling, Linear, Softmax
    model = nn.SpatialAveragePooling(8,8)(model)
    model = nn.Reshape(64)(model)
    model = nn.Linear(64, 10)(model)
    model = nn.LogSoftMax()(model)

    model = nn.gModule({input}, {model})
    model:cuda()
    --print(#model:forward(torch.randn(100, 3, 32,32):cuda()))
else
    print("Loading model from "..opt.loadFrom)
    cutorch.setDevice(1)
    model = torch.load(opt.loadFrom)
    print "Done"
end

loss = nn.ClassNLLCriterion()
loss:cuda()

sgdState = {
   --- For SGD with momentum ---
   ----[[
   -- My semi-working settings
   learningRate   = "will be set later",
   weightDecay    = 1e-4,
   -- Settings from their paper
   --learningRate = 0.1,
   --weightDecay    = 1e-4,

   momentum     = 0.9,
   dampening    = 0,
   nesterov     = true,
   --]]
   --- For rmsprop, which is very fiddly and I don't trust it at all ---
   --[[
   learningRate = "Will be set later",
   alpha = 0.9,
   whichOptimMethod = 'rmsprop',
   --]]
   --- For adadelta, which sucks ---
   --[[
   rho              = 0.3,
   whichOptimMethod = 'adadelta',
   --]]
   --- For adagrad, which also sucks ---
   --[[
   learningRate = "Will be set later",
   whichOptimMethod = 'adagrad',
   --]]
   --- For adam, which also sucks ---
   --[[
   learningRate = 0.005,
   whichOptimMethod = 'adam',
   --]]
   --- For the alternate implementation of NAG ---
   --[[
   learningRate = 0.01,
   weightDecay = 1e-6,
   momentum = 0.9,
   whichOptimMethod = 'nag',
   --]]
   --

   --whichOptimMethod = opt.whichOptimMethod,
}


if opt.loadFrom ~= "" then
    print("Trying to load sgdState from "..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    sgdState = torch.load(""..string.gsub(opt.loadFrom, "model", "sgdState"))
    collectgarbage(); collectgarbage(); collectgarbage()
    print("Got", sgdState.nSampledImages,"images")
end

-- Actual Training! -----------------------------
weights, gradients = model:getParameters()
TrainingHelpers.Init(opt, weights)
function forwardBackwardBatch(checkExitCond)
    -- After every batch, the different GPUs all have different gradients
    -- (because they saw different data), and only the first GPU's weights were
    -- actually updated.
    -- We have to do two changes:
    --   - Copy the new parameters from GPU #1 to the rest of them;
    --   - Zero the gradient parameters so we can accumulate them again.
    model:training()
    gradients:zero()

    --[[
    -- Reset BN momentum, nvidia-style
    model:apply(function(m)
        if torch.type(m):find('BatchNormalization') then
            m.momentum = 1.0  / ((m.count or 0) + 1)
            m.count = (m.count or 0) + 1
            print("--Resetting BN momentum to", m.momentum)
            print("-- Running mean is", m.running_mean:mean(), "+-", m.running_mean:std())
        end
    end)
    --]]

    -- From https://github.com/bgshih/cifar.torch/blob/master/train.lua#L119-L128
    if sgdState.epochCounter < opt.lrDecay1 then
        sgdState.learningRate = 0.1
    elseif sgdState.epochCounter < opt.lrDecay2 then
        sgdState.learningRate = 0.01
    else
        sgdState.learningRate = 0.001
    end

    local loss_val = 0
    local N = opt.iterSize
    local inputs, labels
    for i=1,N do
        inputs, labels = dataTrain:getBatch()
        inputs = inputs:cuda()
        labels = labels:cuda()
        local y = model:forward(inputs)
        loss_val = loss_val + loss:forward(y, labels)
        local df_dw = loss:backward(y, labels)
        model:backward(inputs, df_dw)
    --[[
        collectgarbage(); collectgarbage();
    --]]
        -- The above call will accumulate all GPUs' parameters onto GPU #1
        if checkExitCond and checkExitCond() then
            -- set the real N used
            N = i
            break
        end
    end
    loss_val = loss_val / N
    if N ~= 1 then
      gradients:mul( 1.0 / N )
    end
    -- print("compute batch:", torch.sum(weights), torch.sum(gradients))

    -- lossLog{nImages = sgdState.nSampledImages,
    --        loss = loss_val}

    return loss_val, gradients, inputs:size(1) * N
end

local model_save_dir = opt.saveDir .. "/" .. os.date("%m%d%H%M") .. "/rank_" .. self_rank .. "/"
print("Models will be saved to "..model_save_dir)
os.execute("mkdir -p "..model_save_dir)
local log = assert(io.open(model_save_dir .. "log.txt", "w"))
log:write(pretty_write(opt)..'\n')
log:write(pretty_write(TrainingHelpers.nodes)..'\n')
log:write(pretty_write(TrainingHelpers.weights)..'\n')
log:flush()
function saveModel(epoch, msg)
    model_name = model_save_dir .. epoch .. ".model.t7"
    state_name = model_save_dir .. epoch .. ".state.t7"
    torch.save(model_name, model:clearState())
    torch.save(state_name, sgdState)
    log:write(msg .. "\n")
    log:flush()
end

function evalModel(loss_val, time, average_batch, max_batch)
    loss_val = loss_val or 0
    time = time or 0.0
    average_batch = average_batch or opt.batchSize
    max_batch = max_batch or opt.batchSize
    -- print(string.format("epoch = %d, time = %.3f n_images = %d, avg_batch_size = %.2f, max_batch_size = %.2f, train_loss = %f", 
    --      sgdState.epochCounter or 0, time, sgdState.nSampledImages or 0, average_batch, max_batch, loss_val))
    local results = {loss=0.0, correct1=1.0}
    if not opt.noEval then
        results = evaluateModel(model, loss, dataTest, opt.batchSize)
    end
    local msg = string.format("epoch = %d, time = %.3f, n_images = %d, avg_batch_size = %.2f, max_batch_size = %.2f, train_loss (fake) = %f, test_loss = %f, test_error = %f, learning_rate = %f", 
    sgdState.epochCounter or 0, time, sgdState.nSampledImages or 0, average_batch, max_batch, loss_val, results.loss, 1.0 - results.correct1, sgdState.learningRate)
    print(msg)
    saveModel(sgdState.epochCounter, msg)
    -- print(string.format("epoch = %d, n_images = %d, train_loss (fake) = %f, test_loss = %f, test_error = %f", 
    --      sgdState.epochCounter or 0, sgdState.nSampledImages or 0, loss_val, results.loss, 1.0 - results.correct1))
    --[[ errorLog{nImages = sgdState.nSampledImages or 0,
             error = 1.0 - results.correct1}
    if hasWorkbook then
      if (sgdState.epochCounter or -1) % 10 == 0 then
        workbook:saveTorch("model", model)
        workbook:saveTorch("sgdState", sgdState)
      end
    end
    --]]
end

-- evalModel()

--[[
require 'graph'
graph.dot(model.fg, 'MLP', '/tmp/MLP')
os.execute('convert /tmp/MLP.svg /tmp/MLP.png')
display.image(image.load('/tmp/MLP.png'), {title="Network Structure", win=23})
--]]

--[[
require 'ncdu-model-explore'
local y = model:forward(torch.randn(opt.batchSize, 3, 32,32):cuda())
local df_dw = loss:backward(y, torch.zeros(opt.batchSize):cuda())
model:backward(torch.randn(opt.batchSize,3,32,32):cuda(), df_dw)
exploreNcdu(model)
--]]

-- Begin saving the experiment to our workbook
--[[
if hasWorkbook then
  workbook:saveGitStatus()
  workbook:saveJSON("opt", opt)
end
--]]

-- --[[
TrainingHelpers.trainForever(
forwardBackwardBatch,
weights,
sgdState,
dataTrain:size(),
evalModel
)
--]]
