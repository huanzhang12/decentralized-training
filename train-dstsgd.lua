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


require 'optim'
local DecentralizedSGD = require 'dstsgd'
TrainingHelpers = {}

function TrainingHelpers.Init(opt, params)
   TrainingHelpers.nodes, TrainingHelpers.weights = DecentralizedSGD.LoadConfigFromFile(opt.nodesFile, opt.weightsFile)
   if opt.useCPUforComm == 1 then
      -- this will be used as the buffer for averaging
      TrainingHelpers.cpuParams = params:float()
      params = TrainingHelpers.cpuParams
   end
   TrainingHelpers.dstsgd = DecentralizedSGD.Trainer(TrainingHelpers.nodes, TrainingHelpers.weights, opt.nodeID, {params}, true, opt.chunkSize)
   print("Start init")
   self_rank = TrainingHelpers.dstsgd.Init()
   print("Init done.")
end

function TrainingHelpers.trainForever(forwardBackwardBatch, weights, sgdState, epochSize, afterEpoch)
   local dstsgd = TrainingHelpers.dstsgd
   if sgdState.epochCounter == nil then
      -- first time call this function, will start communication
      dstsgd.StartCommunication()
   end
   sgdState.epochSize = epochSize
   sgdState.epochCounter = sgdState.epochCounter or 0
   sgdState.nSampledImages = sgdState.nSampledImages or 0
   sgdState.nEvalCounter = sgdState.nEvalCounter or 0
   sgdState.thisEpochImages = sgdState.thisEpochImages or 0
   sgdState.thisEpochMaxBatch = sgdState.thisEpochMaxBatch or 0
   sgdState.thisEpochEvalCounter = sgdState.thisEpochEvalCounter or 0
   local whichOptimMethod = optim.sgd
   if sgdState.whichOptimMethod then
       whichOptimMethod = optim[sgdState.whichOptimMethod]
   end
   -- copy the initial weights
   if opt.useCPUforComm == 1 then
      TrainingHelpers.cpuParams:copy(weights)
   end
   local epoch_loss_val = 0.0
   collectgarbage(); collectgarbage()
   timer = torch.Timer()
   while true do -- Each epoch
      local checkExitCond
      if opt.dynBatchSize == 1 then
         checkExitCond = dstsgd.CheckIfClientSyncDone
      else
         checkExitCond = function() return true end
      end
      -- Run forward and backward pass on inputs and labels
      local loss_val, gradients, batchProcessed = forwardBackwardBatch(checkExitCond)
      epoch_loss_val = epoch_loss_val + loss_val
      if opt.dynBatchSize ~= 1 then
         -- wait for this batch, synchronously
         dstsgd.WaitForClientSyncDone()
      end
      -- if the ServerSync is done, we are fine to use the original weights
      local need_keep_param = (not dstsgd.CheckIfServerSyncDone())
      -- got all parameters from peers, averaging!
      -- keep the model parameter being sent, and make a copy
      local new_parameters = dstsgd.AverageParameters(need_keep_param)
      -- print("after average :", need_keep_param, torch.sum(weights), torch.sum(new_parameters["_1"]))
      -- when use CPU for communication, always need to copy the array to GPU
      if opt.useCPUforComm == 1 then
         weights:copy(new_parameters["_1"])
      end
      -- SGD step: modifies weights in-place
      if opt.useCPUforComm == 0 and need_keep_param then
         -- in this case, we need to use new_parameters to update, because weights are still being used
         whichOptimMethod(function() return loss_val, gradients end, new_parameters["_1"], sgdState)
      else
         whichOptimMethod(function() return loss_val, gradients end, weights, sgdState)
      end
      if need_keep_param then
         -- wait for server
         dstsgd.WaitForServerSyncDone()
      end
      if opt.useCPUforComm == 1 then
         -- copy weights back to CPU for communication
         TrainingHelpers.cpuParams:copy(weights)
      end
      if opt.useCPUforComm == 0 and need_keep_param then
         weights:copy(new_parameters["_1"])
      end
      -- weights update done, start next communication iteration
      dstsgd.StartNextIter()
      -- Display progress and loss
      sgdState.nSampledImages = sgdState.nSampledImages + batchProcessed
      sgdState.thisEpochImages = sgdState.thisEpochImages + batchProcessed
      sgdState.thisEpochMaxBatch = math.max(sgdState.thisEpochMaxBatch, batchProcessed)
      sgdState.nEvalCounter = sgdState.nEvalCounter + 1
      sgdState.thisEpochEvalCounter = sgdState.thisEpochEvalCounter + 1
      -- xlua.progress(sgdState.nSampledImages%epochSize, epochSize)
      -- print(string.format("epoch = %d, n_image = %d, train_loss = %f", sgdState.epochCounter, sgdState.nSampledImages, loss_val))
      collectgarbage(); collectgarbage()

      if math.floor(sgdState.nSampledImages / epochSize) ~= sgdState.epochCounter then
         timer:stop()
         collectgarbage(); collectgarbage()
         -- Epoch completed!
         -- xlua.progress(epochSize, epochSize)
         average_batch = sgdState.thisEpochImages / sgdState.thisEpochEvalCounter
         sgdState.epochCounter = math.floor(sgdState.nSampledImages / epochSize)
         epoch_loss_val = epoch_loss_val / dataTrain:size() * opt.batchSize
         if afterEpoch then afterEpoch(epoch_loss_val, timer:time().real, average_batch, sgdState.thisEpochMaxBatch) end
         epoch_loss_val = 0.0
         sgdState.thisEpochImages = 0
         sgdState.thisEpochEvalCounter = 0
         sgdState.thisEpochMaxBatch = 0
         -- print("\n\n----- Epoch "..sgdState.epochCounter.." -----")
         if (sgdState.epochCounter or 0) > opt.maxIter then
            dstsgd.SetExitFlag()
            print("Training complete, go home")
            dstsgd.Terminate()
            os.exit()
         end
         timer:resume()
      end
   end
end


-- Some other stuff that may be helpful but I need to refactor it

-- function TrainingHelpers.inspectLayer(layer, fields)
--    function inspect(x)
--       if x then
--          x = x:double():view(-1)
--          return {
--             p5 = (x:kthvalue(1 + 0.05*x:size(1))[1]),
--             mean = x:mean(),
--             p95 = (x:kthvalue(1 + 0.95*x:size(1))[1]),
--             var = x:var(),
--          }
--       end
--    end
--    local result = {name = tostring(layer)}
--    for _,field in ipairs(fields) do
--       result[field] = inspect(layer[field])
--    end
--    return result
-- end
-- function TrainingHelpers.printLayerInspection(li, fields)
--    print("- "..tostring(li.name))
--    if (string.find(tostring(li.name), "ReLU")
--        or string.find(tostring(li.name), "BatchNorm")
--        or string.find(tostring(li.name), "View")
--        ) then
--        -- Do not print these layers
--    else
--        for _,field in ipairs(fields) do
--           local lf = li[field]
--           if lf then
--               print(string.format(
--                        "%20s    5p: %+3e    Mean: %+3e    95p: %+3e    Var: %+3e",
--                        field, lf.p5, lf.mean, lf.p95, lf.var))
--           end
--        end
--    end
-- end
-- function TrainingHelpers.inspectModel(model)
--    local results = {}
--    for i,layer in ipairs(model.modules) do
--       results[i] = TrainingHelpers.inspectLayer(layer, {"weight",
--                                                         "gradWeight",
--                                                         "bias",
--                                                         "gradBias",
--                                                         "output"})
--    end
--    return results
-- end
-- function TrainingHelpers.printInspection(inspection)
--    print("\n\n\n")
--    print(" \x1b[31m---------------------- Weights ---------------------- \x1b[0m")
--    for i,layer in ipairs(inspection) do
--       TrainingHelpers.printLayerInspection(layer, {"weight", "gradWeight"})
--    end
--    print(" \x1b[31m---------------------- Biases ---------------------- \x1b[0m")
--    for i,layer in ipairs(inspection) do
--       TrainingHelpers.printLayerInspection(layer, {"bias", "gradBias"})
--    end
--    print(" \x1b[31m---------------------- Outputs ---------------------- \x1b[0m")
--    for i,layer in ipairs(inspection) do
--       TrainingHelpers.printLayerInspection(layer, {"output"})
--    end
-- end
-- function displayWeights(model)
--     local layers = {}
--     -- Go through each module and add its weight and its gradient.
--     -- X axis = layer number.
--     -- Y axis = weight / gradient.
--     for i, li in ipairs(model.modules) do
--         if not (string.find(tostring(li), "ReLU")
--             or string.find(tostring(li), "BatchNorm")
--             or string.find(tostring(li), "View")
--             ) then
--             if li.gradWeight then
--                 --print(tostring(li),li.weight:mean())
--                 layers[#layers+1] = {i,
--                     -- Weight
--                     {li.weight:mean() - li.weight:std(),
--                     li.weight:mean(),
--                     li.weight:mean() + li.weight:std()},
--                     -- Gradient
--                     {li.gradWeight:mean() - li.gradWeight:std(),
--                     li.gradWeight:mean(),
--                     li.gradWeight:mean() + li.gradWeight:std()},
--                     -- Output
--                     {li.output:mean() - li.output:std(),
--                     li.output:mean(),
--                     li.output:mean() + li.output:std()},
--                 }
--             end
--         end
--     end
--     -- Plot the result
--     --
--    workbook:plot("Layers", layers, {
--                    labels={"Layer", "Weights", "Gradients", "Outputs"},
--                    customBars=true, errorBars=true,
--                    title='Network Weights',
--                    rollPeriod=1,
--                    win=26,
--                    --annotations={"o"},
--                    --axes={x={valueFormatter="function(x) {return x; }"}},
--              })
-- end
