
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'math'

require 'util.OneHot'
require 'util.misc'

opt = {}
opt.model = 'models/ted_lstm.t7'
opt.seed = 123 
opt.sample = 0
opt.temperature = 2
opt.length = 4
opt.gpuid = 0
opt.opencl = 0
opt.verbose = 0
opt.beam_size = 5

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end
-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- initialize the rnn state to all zeros
--gprint('creating an ' .. checkpoint.opt.model .. '...')

function sample(seed_text, current_state)

if next (current_state) == nil then
  -- table t is empty
    for L = 1,checkpoint.opt.num_layers do
        -- c and h for all layers
        local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
        if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
        table.insert(current_state, h_init:clone())
        if checkpoint.opt.model == 'lstm' then
            table.insert(current_state, h_init:clone())
        end
    end
end
state_size = #current_state


output = ""

-- do a few seeded timesteps
if string.len(seed_text) > 0 then
    gprint('seeding with ' .. seed_text)
    gprint('--------------------------')
    for c in seed_text:gmatch'.' do
        prev_char = torch.Tensor{vocab[c]}
        output = output .. ivocab[prev_char[1]]
        if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
        if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
        local lst = protos.rnn:forward{prev_char, unpack(current_state)}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        current_state = {}
        for i=1,state_size do table.insert(current_state, lst[i]) end
        prediction = lst[#lst] -- last element holds the log probabilities
    end
end

beam = {}
local _, top_indices = prediction:sort(2, true)
for i = 1, opt.beam_size do
    top_char = top_indices[1][i]     
    table.insert(beam, {prediction[1][top_char], seed_text..ivocab[top_char], current_state, opt.length})
end

-- start sampling/argmaxing
while true do
    beam_candidates = {}
    local finish = true
    for _, b in ipairs(beam) do
        -- forward the rnn for next character
        local ch = b[2]:sub(#b[2],#b[2])
        if b[4] > 0 and (ch == ' ' or ch == '.' or ch == '\n' or ch == ',' or ch == '?') then
            b[4] = b[4] - 1
        end
        if b[4] == 0 then
            table.insert(beam_candidates, b)
        else
            finish = false
            prev_char = torch.Tensor{vocab[ch]}
            local lst = protos.rnn:forward{prev_char, unpack(b[3])}
            local current_state = {}
            for i=1,state_size do table.insert(current_state, torch.CudaTensor():resize(lst[i]:size()):copy(lst[i])) end
            prediction = lst[#lst] -- last element holds the log probabilities
            local _, top_indices = prediction:squeeze():sort(1, true)
            for i = 1, opt.beam_size do
                top_char = top_indices[i]
                if top_char > 89 then
                    top_char = 2
                end     
                table.insert(beam_candidates, {b[1] + prediction[1][top_char], b[2]..ivocab[top_char], current_state, b[4]})
            end
        end
    end
    beam = {}
    table.sort(beam_candidates, function(a,b) return a[1] > b[1] end)
    for i = 1, opt.beam_size do
        table.insert(beam, beam_candidates[i])
    end
    if finish then
        break;
    end
end

local output = {}
for _, b in ipairs(beam) do
    table.insert(output, {math.exp(b[1])*100, b[2]})
end
collectgarbage()
return output
end
