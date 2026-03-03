import copy


models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator

# config에서 지정한 모델을 동적으로 생성하고, 필요하면 weight도 불러오는 함수
def make(model_spec, args=None, load_sd=False):
    # config 파일의 args가 비어있지 않다면 복사해옴
    if args is not None:
        model_args = copy.deepcopy(model_spec['args'])
        model_args.update(args)
    else:
    # 만일 새로운 args가 들어온다면 그냥 덮어씀
        model_args = model_spec['args']
    model = models[model_spec['name']](**model_args)
    if load_sd:
        model.load_state_dict(model_spec['sd'])
    return model
