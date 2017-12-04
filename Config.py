from attrdict import AttrDict
def get_config_dict():
    file=open(r'D:\njust\Script\tensorflow\ConvNeuroNet\res\config','r',encoding='utf-8')
    config_dict=eval(file.read())
    return AttrDict(config_dict)
config=get_config_dict()