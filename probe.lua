local Probe, parent = torch.class('nn.Probe', 'nn.Module')

debugger = require('fb.debugger')

function tensorInfo(x, name)
    local name = name or ''
    local sizeStr = ''
    for i = 1, #x:size() do
        sizeStr = sizeStr .. string.format('%d', x:size(i))
        if i < #x:size() then
            sizeStr = sizeStr .. 'x'
        end
    end
    infoStr = string.format('[%15s] size: %12s, min: %+.2e, max: %+.2e', name, sizeStr, x:min(), x:max())
    return infoStr
end

function Probe:__init()    
end

function Probe:updateOutput(input)
    if type(input) == 'table' then
        print(input)
    else
        print(tensorInfo(input, 'input'))
    end
    debugger.enter()
    self.output = input
    return self.output
end

function Probe:updateGradInput(input, gradOutput)
    if type(input) == 'table' then
        print(gradOutput)
    else
        print(tensorInfo(gradOutput, 'gradOutput'))
    end
    self.gradInput = gradOutput
    -- debugger.enter()
    return self.gradInput
end
