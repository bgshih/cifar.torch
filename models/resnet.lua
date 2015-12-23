require 'nn'
require 'cunn'

local backend_name = 'cudnn'

local backend
if backend_name == 'cudnn' then
  require 'cudnn'
  backend = cudnn
else
  backend = nn
end

local resnet = nn.Sequential()

-- Convolutional + SpatialBatchNormalization + ReLU
local function ConvBnRelu(nIn, nOut, k, s, p)
  local block = nn.Sequential()
  block:add(backend.SpatialConvolution(nIn, nOut, k,k, s,s, p,p))
  block:add(nn.SpatialBatchNormalization(nOut, 1e-3))
  block:add(backend.ReLU(true))
  return block
end

-- Residual block contains two ConvBnRelu and a shortcut connection
local function ResBlock(nIn, nOut, subsample)
  local concat
  if subsample then
    local conv = nn.Sequential()
    conv:add(ConvBnRelu(nIn, nOut, 3, 2, 1))
    conv:add(ConvBnRelu(nOut, nOut, 3, 1, 1))
    local shortcut = ConvBnRelu(nIn, nOut, 1, 2, 0)
    concat = nn.ConcatTable():add(conv):add(shortcut)
  else
    local conv = nn.Sequential()
    conv:add(ConvBnRelu(nIn, nOut, 3, 1, 1))
    conv:add(ConvBnRelu(nOut, nOut, 3, 1, 1))
    concat = nn.ConcatTable():add(conv):add(nn.Identity())
  end
  return nn.Sequential():add(concat):add(nn.CAddTable()):add(backend.ReLU(true))
end

local function AddResGroup(nIn, nOut, n, firstSub)
  resnet:add(ResBlock(nIn, nOut, firstSub))
  for i = 1, n-1 do
    resnet:add(ResBlock(nOut, nOut, false))
  end
end

local n = 18
resnet:add(ConvBnRelu(3,16,3,1,1))
AddResGroup(16,16,n,false)
AddResGroup(16,32,n,true)
AddResGroup(32,64,n,true)
resnet:add(backend.SpatialAveragePooling(8,8,1,1,0,0))
resnet:add(nn.Reshape(64))
resnet:add(nn.Linear(64,10))
resnet:add(nn.LogSoftMax())

-- initialization from MSR
local function MSRinit(net)
  local function init(name)
    for k,v in pairs(net:findModules(name)) do
      local n = v.kW*v.kH*v.nOutputPlane
      v.weight:normal(0,math.sqrt(2/n))
      v.bias:zero()
    end
  end
  -- have to do for both backends
  init'cudnn.SpatialConvolution'
  init'nn.SpatialConvolution'
end

MSRinit(resnet)

return resnet
