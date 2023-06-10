import torch
import torch.nn as nn
import json


class Model(nn.Module):
    def __init__(self, module_list: nn.ModuleList) -> None:
        super(Model, self).__init__()
        self.model = module_list

    def forward(self, x):
        for m in self.model:
            x = m(x)

        return x

    def set_input(self, x):
        self.x = x

    def get_input_image_array(self):
        ret = {
            'inputImageArray': self.x.cpu().numpy().tolist()
        }

        return json.dumps(ret)

    @torch.no_grad()
    def get_feature_map(self):
        self.model.eval()
        assert self.x is not None
        x = self.x

        output = []
        for m in self.model:
            x = m(x)
            output.append(x.cpu().numpy().tolist())

        self.model.train()

        output = {'allOutputs': output}
        output = json.dumps(output)
        return output

    @torch.no_grad()
    def get_model(self):
        ret = []
        global_id = 0
        local_id = 1
        for m in self.model:
            name = str(type(m))
            if 'conv' in name:
                global_id += 1
                local_id = 1

                param = dict(m.named_parameters())
                param['kernel'] = param.pop('weight').cpu().numpy().tolist()
                param['bias'] = param.pop('bias').cpu().numpy().tolist()
                param['name'] = f'conv_{global_id}_{local_id}'
            elif 'activation' in name:
                local_id += 1
                param = {
                    'name': f'relu_{global_id}_{local_id}'
                }
            elif 'pooling' in name:
                local_id += 1
                param = {
                    'name': f'pool_{global_id}_{local_id}'
                }
            elif 'Flatten' in name:
                global_id += 1
                local_id = 1

                param = {
                    'name': f'flatten_{global_id}_{local_id}'
                }
            elif 'linear' in name:
                global_id += 1
                local_id = 1

                param = dict(m.named_parameters())
                param['kernel'] = param.pop('weight').cpu().numpy().tolist()
                param['bias'] = param.pop('bias').cpu().numpy().tolist()
                param['name'] = f'output_{global_id}_{local_id}'
            else:
                raise ValueError('Invalid layer.')

            ret.append(param)

        ret = {'layers': ret}
        ret = json.dumps(ret)
        return ret


class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x)


def construct(struct_str):
    struct = struct_str.split("/")
    module_list = []
    size = 64
    for idx, module_name in enumerate(struct):
        if module_name == 'conv':
            module = []
            if idx == 0:
                module.append(nn.Conv2d(3, 10, 3))
            else:
                module.append(nn.Conv2d(10, 10, 3))
            module.append(nn.ReLU())
            size -= 2
        elif module_name == 'pool':
            module = [nn.MaxPool2d(2)]
            size //= 2
        else:
            raise ValueError('Invalid layer name.')
        
        module_list.extend(module)

    module_list.append(Flatten())
    module_list.append(
        nn.Linear(10 * size * size, 10, bias=True)
    )

    module_list = nn.ModuleList(module_list)
    model = Model(module_list)
    return model
    

img = torch.ones((1, 3, 64, 64))
struct = "conv/conv/pool/conv/conv/pool"
model = construct(struct)
model.set_input(img)
# print(model.get_model())

#for o in output:
#    print(o.shape)

# for p in param:
#     for k, v in p.items():
#         if 'name' in k:
#             print(k, v)
#         elif 'kernel' in k or 'bias' in k:
#             print(k, v.shape)
