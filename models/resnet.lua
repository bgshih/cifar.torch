require('nn')
require('cunn')
require('cudnn')


local backend = cudnn

local resnet = nn.Sequential()

-- a residual block contains two convs and a shortcut
local function ResBlock(nIn, nOut, subsample)
  local block = nn.Sequential()
  local concat
  if subsample then
    local conv = nn.Sequential()
    conv:add(backend.SpatialConvolution(nIn, nOut, 3,3, 2,2, 1,1))
    conv:add(nn.SpatialBatchNormalization(nOut, 1e-3))
    conv:add(backend.ReLU(true))
    conv:add(backend.SpatialConvolution(nOut, nOut, 3,3, 1,1, 1,1))
    local shortcut = backend.SpatialConvolution(nIn, nOut, 1,1, 2,2, 0,0)
    concat = nn.ConcatTable():add(conv):add(shortcut)
  else
    local conv = nn.Sequential()
    conv:add(backend.SpatialConvolution(nIn, nOut, 3,3, 1,1, 1,1))
    conv:add(nn.SpatialBatchNormalization(nOut, 1e-3))
    conv:add(backend.ReLU(true))
    conv:add(backend.SpatialConvolution(nOut, nOut, 3,3, 1,1, 1,1))
    concat = nn.ConcatTable():add(conv):add(nn.Identity())
  end
  block:add(concat)
  block:add(nn.CAddTable())
  block:add(backend.ReLU(true))
  return block
end

-- a residual group contains n residual blocks
local function ResGroup(nIn, nOut, n, firstSub)
  local group = nn.Sequential()
  group:add(ResBlock(nIn, nOut, firstSub))
  for i = 1, n-1 do
    group:add(ResBlock(nOut, nOut, false))
  end
  return group
end

local n = 18
resnet:add(backend.SpatialConvolution(3, 16, 3,3, 1,1, 1,1))
resnet:add(nn.SpatialBatchNormalization(16, 1e-3))
resnet:add(backend.ReLU(true))
resnet:add(ResGroup(16,16,n,false))
resnet:add(ResGroup(16,32,n,true))
resnet:add(ResGroup(32,64,n,true))
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
