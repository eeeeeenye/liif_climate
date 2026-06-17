import copy

# 등록된 dataset 클래스들을 저장하는 딕셔너리
# # datasets = {
#     'image-folder': ImageFolder,
#     'sr-implicit': SRImplicitDataset
# }
datasets = {}


def register(name):
    """
    Dataset 클래스를 registry에 등록하기 위한 decorator

    @register('image-folder')
    class ImageFolder(...):
        ...

    그러면
    datasets['image-folder'] = ImageFolder
    가 자동으로 수행됨.
    """

    def decorator(cls):

        # datasets 딕셔너리에
        # 이름(name) -> 클래스(cls) 저장
        datasets[name] = cls

        return cls

    return decorator


def make(dataset_spec, args=None):
    """
    dataset_spec 정보를 이용해 dataset 객체 생성

    dataset_spec = {
        'name': 'image-folder',
        'args': {
            'root_path': './data'
        }
    }

    -> ImageFolder(root_path='./data')
    생성
    """

    if args is not None:

        # 원본 config를 건드리지 않기 위해 deepcopy
        dataset_args = copy.deepcopy(
            dataset_spec['args']
        )

        # 추가 인자가 있으면 덮어쓰기
        dataset_args.update(args)

    else:

        # config에 정의된 기본 args 사용
        dataset_args = dataset_spec['args']

    # registry에서 클래스 찾아 생성
    #
    # 예:
    # datasets['image-folder']
    # -> ImageFolder 클래스
    #
    # ImageFolder(**dataset_args)
    dataset = datasets[
        dataset_spec['name']
    ](**dataset_args)

    return dataset